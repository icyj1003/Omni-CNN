import os
import torch
from tqdm import tqdm
import copy
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _run_single_client_round(
    args,
    client,
    round_idx,
    model_common_curr_params,
    lidar_model_curr_params,
    img_model_curr_params,
    gps_model_curr_params,
):
    client.update_model(
        model_common_curr_params,
        lidar_model_curr_params,
        img_model_curr_params,
        gps_model_curr_params,
    )
    print("Client {} is inferencing...".format(client.client_id))
    infer_prec1 = client.model_testing_on_local_data()

    if round_idx == 0:
        client.update_previous_accuracy(infer_prec1)

    is_transfer = evaluate_accuracy(args, client, infer_prec1)
    if is_transfer:
        client.set_transfer_learning()
        overhead = client.transfer_overhead
        print("Client {} is transferring...".format(client.client_id))
    else:
        client.set_relearning()
        overhead = client.overhead
        print("Client {} is retraining...".format(client.client_id))

    _common_mask = client.get_mask("common")
    _lidar_mask = client.get_mask("lidar")
    _img_mask = client.get_mask("img")
    _gps_mask = client.get_mask("gps")

    print(
        f"Checking mask... common_mask: {_common_mask is not None}, lidar_mask: {_lidar_mask is not None}, img_mask: {_img_mask is not None}, gps_mask: {_gps_mask is not None}"
    )

    client.client_local_training(5)
    prec1 = client.model_testing_on_local_data()
    delta_acc = prec1 - infer_prec1
    client.update_delta_acc(delta_acc)

    return {
        "client": client,
        "infer_prec1": infer_prec1,
        "prec1": prec1,
        "delta_acc": delta_acc,
        "is_transfer": is_transfer,
        "overhead": overhead,
        "common_mask": _common_mask,
        "lidar_mask": _lidar_mask,
        "img_mask": _img_mask,
        "gps_mask": _gps_mask,
        "model_common_state": copy.deepcopy(client.model_common.state_dict()),
        "lidar_state": copy.deepcopy(client.lidar_model.state_dict())
        if _lidar_mask is not None
        else None,
        "img_state": copy.deepcopy(client.img_model.state_dict())
        if _img_mask is not None
        else None,
        "gps_state": copy.deepcopy(client.gps_model.state_dict())
        if _gps_mask is not None
        else None,
    }


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
    size_limits = [x * 1e6 for x in [3.4, 2.2, 2.5, 2.0, 3.1, 2.6, 2.9, 2.4, 2.1, 1.9]]
    list_of_clients = []
    common_total_size = 0
    lidar_total_size = 0
    img_total_size = 0
    gps_total_size = 0

    for i in args.clients:
        random_key = equipment_list[int(i)]
        print("Client {} has {}".format(i, random_key))
        client_data_path = os.path.join(args.base_path, "Client_" + str(i))
        client_save_path = os.path.join(args.save_path, args.name, "Client_" + str(i))
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

        # AvgFed counters (used only when args.use_tfed == False)
        common_client_count = 0
        lidar_client_count = 0
        img_client_count = 0
        gps_client_count = 0

        max_client_workers = max(1, int(getattr(args, "client_train_workers", 1)))
        if max_client_workers == 1 or len(list_of_clients) == 1:
            client_round_results = [
                _run_single_client_round(
                    args,
                    client,
                    i,
                    model_common_curr_params,
                    lidar_model_curr_params,
                    img_model_curr_params,
                    gps_model_curr_params,
                )
                for client in list_of_clients
            ]
        else:
            print(
                "Running client training in parallel with {} workers".format(
                    max_client_workers
                )
            )
            client_round_results = []
            with ThreadPoolExecutor(max_workers=max_client_workers) as executor:
                futures = {
                    executor.submit(
                        _run_single_client_round,
                        args,
                        client,
                        i,
                        model_common_curr_params,
                        lidar_model_curr_params,
                        img_model_curr_params,
                        gps_model_curr_params,
                    ): client
                    for client in list_of_clients
                }
                for future in as_completed(futures):
                    client_round_results.append(future.result())

        client_round_results.sort(key=lambda x: int(x["client"].client_id))

        for result in client_round_results:
            client = result["client"]
            infer_prec1 = result["infer_prec1"]
            prec1 = result["prec1"]
            delta_acc = result["delta_acc"]
            _common_mask = result["common_mask"]
            _lidar_mask = result["lidar_mask"]
            _img_mask = result["img_mask"]
            _gps_mask = result["gps_mask"]

            top1_infer.update(infer_prec1, 1)
            WRITER.add_scalar(
                "Client_"
                + str(client.client_id)
                + "/InferenceAccuracy_"
                + "_".join(client.equipment),
                infer_prec1,
                i,
            )

            if args.use_tfed:
                overhead = result["overhead"]
                WRITER.add_scalar(
                    "Client_"
                    + str(client.client_id)
                    + "/Is_Transfer_Learning_"
                    + "_".join(client.equipment),
                    result["is_transfer"],
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
            else:
                WRITER.add_scalar(
                    "Client_"
                    + str(client.client_id)
                    + "/Overhead_total_"
                    + "_".join(client.equipment),
                    client.avg_overhead,
                    i + 1,
                )

            WRITER.add_scalar(
                "Client_"
                + str(client.client_id)
                + "/Accuracy_"
                + "_".join(client.equipment),
                prec1,
                i + 1,
            )
            top1.update(prec1, 1)

            WRITER.add_scalar(
                "Client_"
                + str(client.client_id)
                + "/Delta_Acc_"
                + "_".join(client.equipment),
                delta_acc,
                i + 1,
            )

            common_state = result["model_common_state"]
            for name, param in common_state.items():
                if args.use_tfed:
                    model_common_new_params[name] += param * (
                        _common_mask[name].cuda() * client.get_train_size()
                    )
                    cummulative_common_mask[name] += (
                        _common_mask[name].cuda() * client.get_train_size()
                    )
                else:
                    # plain average (equal weight per client)
                    model_common_new_params[name] += param
            if not args.use_tfed:
                common_client_count += 1

            if _lidar_mask is not None and result["lidar_state"] is not None:
                for name, param in result["lidar_state"].items():
                    if args.use_tfed:
                        lidar_weights = sigmoid_with_zero_handling(
                            _lidar_mask[name].cuda()
                            * client.get_delta_acc()
                            * client.get_train_size()
                        )
                        lidar_model_new_params[name] += param * lidar_weights
                        cummulative_lidar_mask[name] += lidar_weights
                    else:
                        lidar_model_new_params[name] += param
                if not args.use_tfed:
                    lidar_client_count += 1

            if _img_mask is not None and result["img_state"] is not None:
                for name, param in result["img_state"].items():
                    if args.use_tfed:
                        img_weights = sigmoid_with_zero_handling(
                            _img_mask[name].cuda()
                            * client.get_delta_acc()
                            * client.get_train_size()
                        )
                        img_model_new_params[name] += param * img_weights
                        cummulative_img_mask[name] += img_weights
                    else:
                        img_model_new_params[name] += param
                if not args.use_tfed:
                    img_client_count += 1

            if _gps_mask is not None and result["gps_state"] is not None:
                for name, param in result["gps_state"].items():
                    if args.use_tfed:
                        gps_weights = sigmoid_with_zero_handling(
                            _gps_mask[name].cuda()
                            * client.get_delta_acc()
                            * client.get_train_size()
                        )
                        gps_model_new_params[name] += param * gps_weights
                        cummulative_gps_mask[name] += gps_weights
                    else:
                        gps_model_new_params[name] += param
                if not args.use_tfed:
                    gps_client_count += 1

            # client log here

        print("Round {} : Average Accuracy: {}".format(i + 1, top1.avg))
        print("Round {} : Average Inference Accuracy: {}".format(i + 1, top1_infer.avg))

        # log to tensorboard
        WRITER.add_scalar("AverageAccuracy", top1.avg, i + 1)
        WRITER.add_scalar("AverageInferenceAccuracy", top1_infer.avg, i + 1)

        if args.use_tfed:
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
        else:
            # plain AvgFed (equal weight per client)
            for name in model_common_new_params:
                model_common_new_params[name] = model_common_new_params[name] / max(
                    common_client_count, 1
                )

            for name in lidar_model_new_params:
                if lidar_client_count > 0:
                    lidar_model_new_params[name] = (
                        lidar_model_new_params[name] / lidar_client_count
                    )
                else:
                    lidar_model_new_params[name] = lidar_model_curr_params[name]

            for name in img_model_new_params:
                if img_client_count > 0:
                    img_model_new_params[name] = (
                        img_model_new_params[name] / img_client_count
                    )
                else:
                    img_model_new_params[name] = img_model_curr_params[name]

            for name in gps_model_new_params:
                if gps_client_count > 0:
                    gps_model_new_params[name] = (
                        gps_model_new_params[name] / gps_client_count
                    )
                else:
                    gps_model_new_params[name] = gps_model_curr_params[name]

        print("Server is updating...")

        model_common.load_state_dict(model_common_new_params)
        lidar_model.load_state_dict(lidar_model_new_params)
        img_model.load_state_dict(img_model_new_params)
        gps_model.load_state_dict(gps_model_new_params)

    # test the global model on joint test set
    for client in list_of_clients:
        client.save_model()

    return model_common, lidar_model, img_model, gps_model
