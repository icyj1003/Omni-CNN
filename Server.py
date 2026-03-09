import os
import torch
from tqdm import tqdm
import copy
import random
from Clients import Client_pipeline, AverageMeter

seed = 50
random.seed(seed)


from datetime import datetime
import math


gps_accuracy_threshold = 29.3
img_accuracy_threshold = 71
lidar_accuracy_threshold = 85
gps_img_accuracy_threshold = 68
gps_lidar_accuracy_threshold = 85
img_lidar_accuracy_threshold = 87
gps_img_lidar_accuracy_threshold = 88


def sigmoid_with_zero_handling(tensor):
    """
    Computes the sigmoid function element-wise for a PyTorch tensor,
    with zero input resulting in zero output.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor after applying the sigmoid function.
    """
    # Apply sigmoid only where the tensor is non-zero
    sigmoid_result = torch.where(
        tensor != 0, 1 / (1 + torch.exp(-tensor)), torch.tensor(0.0)
    )
    return sigmoid_result


def count_nonzero_parameters(model):
    return sum(torch.count_nonzero(param).item() for param in model.parameters())


def count_nonzero_mask(mask):
    return sum([v.sum().item() for v in mask.values()])


def safe_elementwise_division(tensor1, tensor2):
    """
    Performs element-wise division of two tensors with zero elements in the denominator
    resulting in a zero output.

    Args:
        tensor1 (torch.Tensor): Numerator tensor.
        tensor2 (torch.Tensor): Denominator tensor.

    Returns:
        torch.Tensor: Result of element-wise division.
    """
    # Replace zeros in the denominator with ones to avoid division by zero
    result = torch.where(
        tensor2 != 0, tensor1.cuda() / tensor2.cuda(), torch.tensor(0.0)
    )
    return result.cuda()


def evaluate_accuracy(args, client, accuracy):
    if (
        "gps" in client.equipment
        and "img" in client.equipment
        and "lidar" in client.equipment
    ):
        if accuracy > gps_img_lidar_accuracy_threshold:
            return True
    elif "gps" in client.equipment and "img" in client.equipment:
        if accuracy > gps_img_accuracy_threshold:
            return True
    elif "gps" in client.equipment and "lidar" in client.equipment:
        if accuracy > gps_lidar_accuracy_threshold:
            return True
    elif "img" in client.equipment and "lidar" in client.equipment:
        if accuracy > img_lidar_accuracy_threshold:
            return True
    elif "gps" in client.equipment:
        if accuracy > gps_accuracy_threshold:
            return True
    elif "img" in client.equipment:
        if accuracy > img_accuracy_threshold:
            return True
    elif "lidar" in client.equipment:
        if accuracy > lidar_accuracy_threshold:
            return True
    else:
        raise ValueError("Invalid equipment combination")
    return False


def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False


def federated_train(
    args,
    model_common,
    lidar_model,
    lidar_mask,
    img_model,
    img_mask,
    gps_model,
    gps_mask,
    WRITER,
):

    equipment_list = [
        ["lidar", "img", "gps"],
        ["lidar"],
        ["img"],
        ["gps"],
        ["lidar", "img"],
        ["lidar", "gps"],
        ["img", "gps"],
        ["lidar", "img", "gps"],
        ["lidar", "img"],
        ["lidar", "gps"],
    ]
    size_limits = [
        x * 1e6 if args.use_tfed else 1e8
        for x in [3.4, 2.2, 2.5, 2.0, 3.1, 2.6, 2.9, 2.4, 2.1, 1.9]
    ]
    list_of_clients = []
    common_total_size = 0
    lidar_total_size = 0
    img_total_size = 0
    gps_total_size = 0

    for i in args.clients:
        random_key = equipment_list[int(i)]
        print("Client {} has {}".format(i, random_key))
        client_data_path = os.path.join(args.base_path, "Client_" + str(i))
        client_save_path = os.path.join(args.save_path, "Client_" + str(i))
        client_pipeline = Client_pipeline(
            args,
            client_data_path,
            client_save_path,
            i,
            random_key,
        )
        train_size = client_pipeline.load_data()
        common_total_size += train_size
        if "lidar" in random_key:
            lidar_total_size += train_size
        if "img" in random_key:
            img_total_size += train_size
        if "gps" in random_key:
            gps_total_size += train_size

        client_pipeline.load_model(
            model_common,
            lidar_model,
            lidar_mask,
            img_model,
            img_mask,
            gps_model,
            gps_mask,
            size_limits[int(i)],
        )

        client_pipeline.configure_optimizer()
        list_of_clients.append(client_pipeline)
    for i in tqdm(range(args.comms_round)):
        top1 = AverageMeter()
        print("Start Round {} ...".format(i + 1))
        top1_infer = AverageMeter()
        model_common_curr_params = copy.deepcopy(model_common.state_dict())
        lidar_model_curr_params = copy.deepcopy(lidar_model.state_dict())
        img_model_curr_params = copy.deepcopy(img_model.state_dict())
        gps_model_curr_params = copy.deepcopy(gps_model.state_dict())
        model_common_new_params = dict(
            [(name, 0) for name, param in model_common_curr_params.items()]
        )
        lidar_model_new_params = dict(
            [(name, 0) for name, param in lidar_model_curr_params.items()]
        )
        img_model_new_params = dict(
            [(name, 0) for name, param in img_model_curr_params.items()]
        )
        gps_model_new_params = dict(
            [(name, 0) for name, param in gps_model_curr_params.items()]
        )
        cummulative_common_mask = dict(
            [(name, 0) for name, param in model_common_curr_params.items()]
        )
        cummulative_lidar_mask = dict(
            [(name, 0) for name, param in lidar_model_curr_params.items()]
        )
        cummulative_img_mask = dict(
            [(name, 0) for name, param in img_model_curr_params.items()]
        )
        cummulative_gps_mask = dict(
            [(name, 0) for name, param in gps_model_curr_params.items()]
        )
        for client in list_of_clients:
            client.update_model(
                model_common_curr_params,
                lidar_model_curr_params,
                img_model_curr_params,
                gps_model_curr_params,
            )
            print("Client {} is inferencing...".format(client.client_id))
            infer_prec1 = client.model_testing_on_local_data()
            top1_infer.update(infer_prec1, 1)
            WRITER.add_scalar(
                "Client_"
                + str(client.client_id)
                + "/InferenceAccuracy_"
                + "_".join(client.equipment),
                infer_prec1,
                i,
            )
            if i == 0:
                client.update_previous_accuracy(infer_prec1)
            if evaluate_accuracy(args, client, infer_prec1):
                client.set_transfer_learning()
                overhead = client.transfer_overhead
                print("Client {} is transferring...".format(client.client_id))
            else:
                client.set_relearning()
                overhead = client.overhead
                print("Client {} is retraining...".format(client.client_id))

            WRITER.add_scalar(
                "Client_"
                + str(client.client_id)
                + "/Is_Transfer_Learning_"
                + "_".join(client.equipment),
                evaluate_accuracy(args, client, infer_prec1),
                i + 1,
            )
            WRITER.add_scalar(
                "Client_"
                + str(client.client_id)
                + "/Overhead_weight_"
                + "_".join(client.equipment),
                overhead[1],
                i + 1,
            )
            WRITER.add_scalar(
                "Client_"
                + str(client.client_id)
                + "/Overhead_mask_"
                + "_".join(client.equipment),
                overhead[2],
                i + 1,
            )
            WRITER.add_scalar(
                "Client_"
                + str(client.client_id)
                + "/Overhead_total_"
                + "_".join(client.equipment),
                sum(overhead),
                i + 1,
            )
            _common_mask = client.get_mask("common")
            _lidar_mask = client.get_mask("lidar")
            _img_mask = client.get_mask("img")
            _gps_mask = client.get_mask("gps")

            print(
                f"Checking mask... common_mask: {_common_mask is not None}, lidar_mask: {_lidar_mask is not None}, img_mask: {_img_mask is not None}, gps_mask: {_gps_mask is not None}"
            )

            client.client_local_training(5)

            prec1 = client.model_testing_on_local_data()
            WRITER.add_scalar(
                "Client_"
                + str(client.client_id)
                + "/Accuracy_"
                + "_".join(client.equipment),
                prec1,
                i + 1,
            )
            top1.update(prec1, 1)

            delta_acc = prec1 - infer_prec1
            WRITER.add_scalar(
                "Client_"
                + str(client.client_id)
                + "/Delta_Acc_"
                + "_".join(client.equipment),
                delta_acc,
                i + 1,
            )

            client.update_delta_acc(delta_acc)

            for name, param in client.model_common.state_dict().items():
                # sigmoid_coeff = sigmoid_with_zero_handling(
                #     _common_mask[name].cuda() * client.get_delta_acc()
                # )
                # model_common_new_params[name] += param * sigmoid_coeff
                # cummulative_common_mask[name] += sigmoid_coeff
                model_common_new_params[name] += param * (
                    _common_mask[name].cuda() * client.get_train_size()
                )
                cummulative_common_mask[name] += (
                    _common_mask[name].cuda() * client.get_train_size()
                )
            if _lidar_mask is not None:
                for name, param in client.lidar_model.state_dict().items():
                    # sigmoid_coeff = sigmoid_with_zero_handling(
                    #     _lidar_mask[name].cuda() * client.get_delta_acc()
                    # )
                    # lidar_model_new_params[name] += param * sigmoid_coeff
                    # cummulative_lidar_mask[name] += sigmoid_coeff
                    lidar_model_new_params[name] += param * (
                        _lidar_mask[name].cuda() * client.get_train_size()
                    )
                    cummulative_lidar_mask[name] += (
                        _lidar_mask[name].cuda() * client.get_train_size()
                    )
            if _img_mask is not None:
                for name, param in client.img_model.state_dict().items():
                    # sigmoid_coeff = sigmoid_with_zero_handling(
                    #     _img_mask[name].cuda() * client.get_delta_acc()
                    # )
                    # img_model_new_params[name] += param * sigmoid_coeff
                    # cummulative_img_mask[name] += sigmoid_coeff
                    img_model_new_params[name] += param * (
                        _img_mask[name].cuda() * client.get_train_size()
                    )
                    cummulative_img_mask[name] += (
                        _img_mask[name].cuda() * client.get_train_size()
                    )
            if _gps_mask is not None:
                for name, param in client.gps_model.state_dict().items():
                    # sigmoid_coeff = sigmoid_with_zero_handling(
                    #     _gps_mask[name].cuda() * client.get_delta_acc(),
                    # )
                    # gps_model_new_params[name] += param * sigmoid_coeff
                    # cummulative_gps_mask[name] += sigmoid_coeff
                    gps_model_new_params[name] += param * (
                        _gps_mask[name].cuda() * client.get_train_size()
                    )
                    cummulative_gps_mask[name] += (
                        _gps_mask[name].cuda() * client.get_train_size()
                    )

            # client log here

        print("Round {} : Average Accuracy: {}".format(i + 1, top1.avg))
        print("Round {} : Average Inference Accuracy: {}".format(i + 1, top1_infer.avg))

        # log to tensorboard
        WRITER.add_scalar("AverageAccuracy", top1.avg, i + 1)
        WRITER.add_scalar("AverageInferenceAccuracy", top1_infer.avg, i + 1)

        # aggregate global model parameters
        for name, param in model_common_new_params.items():
            model_common_new_params[name] = safe_elementwise_division(
                param, cummulative_common_mask[name]
            )
        for name, param in lidar_model_new_params.items():
            try:
                if torch.sum(param) != 0:
                    print("updating lidar model")
                    lidar_model_new_params[name] = safe_elementwise_division(
                        param, cummulative_lidar_mask[name]
                    )
                else:
                    print("not updating lidar model")
                    lidar_model_new_params[name] = lidar_model_curr_params[name]
            except Exception:
                print(f"Checking lidar model param: {param} so not updating")
                lidar_model_new_params[name] = lidar_model_curr_params[name]
        for name, param in img_model_new_params.items():
            try:
                if torch.sum(param) != 0:
                    print("updating img model")
                    img_model_new_params[name] = safe_elementwise_division(
                        param, cummulative_img_mask[name]
                    )
                else:
                    print("not updating img model")
                    img_model_new_params[name] = img_model_curr_params[name]
            except Exception:
                print(f"Checking img model param: {param} so not updating")
                img_model_new_params[name] = img_model_curr_params[name]
        for name, param in gps_model_new_params.items():
            try:
                if torch.sum(param) != 0:
                    print("updating gps model")
                    gps_model_new_params[name] = safe_elementwise_division(
                        param, cummulative_gps_mask[name]
                    )
                else:
                    print("not updating gps model")
                    gps_model_new_params[name] = gps_model_curr_params[name]
            except Exception:
                print(f"Checking gps model param: {param} so not updating")
                gps_model_new_params[name] = gps_model_curr_params[name]

        print("Server is updating...")

        model_common.load_state_dict(model_common_new_params)
        lidar_model.load_state_dict(lidar_model_new_params)
        img_model.load_state_dict(img_model_new_params)
        gps_model.load_state_dict(gps_model_new_params)
