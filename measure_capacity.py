from pyexpat import model

import torch
import os
import pickle


def measure_modelsize(model_path: str):
    """
    Loads a PyTorch model from the given file path, computes:
      1. The total number of non-zero parameters.
      2. The approximate "sparse" model volume in MB
         (assuming 4 bytes per non-zero parameter).
      3. The actual file size on disk in MB.

    Args:
        model_path (str): Path to the .pt (or .pth) file containing the model.

    Returns:
        tuple: (non_zero_params, approx_sparse_model_volume_mb, file_size_mb)
    """

    # Load the state_dict (collections.OrderedDict)
    # Load the state_dict (collections.OrderedDict)
    state_dict = torch.load(model_path, map_location="cpu")

    # If your .pt file has a different structure (e.g. wrapped under state_dict['model']),
    # adjust how you access the parameters accordingly.

    total_nonzero = 0
    total_zero = 0

    # Iterate over each parameter tensor in the OrderedDict
    for key, tensor_value in state_dict.items():
        # if isinstance(tensor_value, torch.Tensor) and key in ["out.weight", "out.bias"]:
        nonzero_count = (tensor_value != 0).sum().item()
        zero_count = tensor_value.numel() - nonzero_count
        total_nonzero += nonzero_count
        total_zero += zero_count
        # else:
        #     total_zero += tensor_value.numel()
        #     print(f"Skipping {key} and total_zero is {total_zero}")

    # Calculate total bits:
    #   Non-zero parameters: float16 => 16 bits
    #   Zero parameters: 1 bit
    total_bits = (
        total_nonzero * 16 + total_zero * 1 + (total_nonzero + total_zero) * 1 + 16
    )

    # Convert bits -> bytes -> MB
    #   1 byte = 8 bits
    #   1 MB = 1024 * 1024 bytes
    model_size_mb = total_bits / 8 / (1024**2)

    return total_nonzero, total_zero, total_bits, model_size_mb


def compute_model_size_in_mb(state_dict):
    """
    Compute the size of a model (in MB) under the assumption:
      - non-zero parameters are float16 (16 bits)
      - zero parameters are 1 bit
    """
    total_nonzero = 0
    total_zero = 0

    for param in state_dict.values():
        if isinstance(param, torch.Tensor):
            nonzero_count = (param != 0).sum().item()
            zero_count = param.numel() - nonzero_count
            total_nonzero += nonzero_count
            total_zero += zero_count

    # total bits = (non-zero * 16) + (zero * 1)
    total_bits = total_nonzero * 16 + total_zero
    # bits -> bytes -> MB
    total_mb = total_bits / 8 / (1024**2)
    return total_mb


def one_shot_prune_to_size(
    model_path: str,
    max_size_mb: float,
    pruned_model_path: str = None,
    mask_path: str = None,
):
    """
    One-shot pruning of the model (OrderedDict of tensors). The threshold is chosen
    so that the final size is <= `max_size_mb` if possible. We assume:
       * Non-zero params = float16 (16 bits)
       * Zero params = 1 bit

    We also save the pruning mask as a .pkl file if `mask_path` is provided.

    :param model_path: Path to the .pt file containing the model's state_dict.
    :param max_size_mb: The maximum allowed model size in MB.
    :param pruned_model_path: If not None, save the pruned model to this path.
    :param mask_path: If not None, save the binary pruning mask to this .pkl file.
    :return: (pruned_state_dict, final_size_mb, threshold)
    """

    # 1. Load the state_dict (OrderedDict).
    state_dict = torch.load(model_path, map_location="cpu")
    # If your model has a different structure (e.g., state_dict["model"]), adjust as needed.

    # 2. Flatten all parameters (absolute values) in a single 1D tensor.
    all_params = []
    for param in state_dict.values():
        if isinstance(param, torch.Tensor) and torch.is_floating_point(param):
            all_params.append(
                param.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            )

    if not all_params:
        raise ValueError("No floating-point tensors found in the state_dict.")

    all_params = torch.cat(all_params, dim=0)
    abs_values = all_params.abs()
    total_num_params = abs_values.numel()

    # 3. Compute how many parameters we can keep non-zero to fit within max_size_mb.
    #    Convert MB -> bits
    user_bits_limit = max_size_mb * (1024**2) * 8

    # size in bits = (#nonzero * 16) + (#zero * 1)
    #              = 16*#nonzero + 1*(N - #nonzero)
    #              = 15*#nonzero + N
    # We want: 15*#nonzero + N <= user_bits_limit
    # =>       #nonzero <= (user_bits_limit - N) / 15
    max_nonzero_possible = (user_bits_limit - total_num_params) / 15.0
    max_nonzero_possible = int(max(0, min(total_num_params, max_nonzero_possible)))

    # 4. Determine the threshold.
    if max_nonzero_possible == 0:
        # We must set everything to zero
        threshold = float("inf")
    elif max_nonzero_possible == total_num_params:
        # We can keep everything
        threshold = 0.0
    else:
        # Sort in descending order to find the value at the (max_nonzero_possible)-th position
        sorted_values, _ = torch.sort(abs_values, descending=True)
        threshold = sorted_values[max_nonzero_possible - 1].item()

    # 5. Prune parameters
    pruned_state_dict = {}
    pruning_mask_dict = {}
    for key, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            pruned_param = param.clone()
            # Create a mask: 1 where |value| >= threshold, 0 otherwise
            mask = pruned_param.abs() >= threshold
            # Zero out values below threshold
            pruned_param[~mask] = 0

            pruned_state_dict[key] = pruned_param
            pruning_mask_dict[key] = mask
        else:
            pruned_state_dict[key] = param

    # 6. Compute final size
    final_size_mb = compute_model_size_in_mb(pruned_state_dict)

    # 7. Optionally save
    if pruned_model_path is not None:
        torch.save(pruned_state_dict, pruned_model_path)

    if mask_path is not None:
        with open(mask_path, "wb") as f:
            pickle.dump(pruning_mask_dict, f)

    return pruned_state_dict, final_size_mb, threshold


def one_shot_prune_to_param_limit(
    state_dict,
    max_params: float,
):

    # 2. Flatten all parameters (absolute values) in a single 1D tensor.
    all_params = []
    for param in state_dict.values():
        if isinstance(param, torch.Tensor) and torch.is_floating_point(param):
            all_params.append(
                param.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            )

    if not all_params:
        raise ValueError("No floating-point tensors found in the state_dict.")

    all_params = torch.cat(all_params, dim=0)
    abs_values = all_params.abs()
    total_num_params = abs_values.numel()

    # 3. Compute how many parameters we can keep non-zero to fit within max_params.
    max_nonzero_params = int(min(max_params, total_num_params))

    # 4. Determine the threshold.
    if max_nonzero_params == 0:
        # We must set everything to zero
        threshold = float("inf")
    elif max_nonzero_params == total_num_params:
        # We can keep everything
        threshold = 0.0
    else:
        # Sort in descending order to find the value at the (max_nonzero_params)-th position
        sorted_values, _ = torch.sort(abs_values, descending=True)
        threshold = sorted_values[max_nonzero_params - 1].item()

    # 5. Prune parameters
    pruned_state_dict = {}
    pruning_mask_dict = {}
    for key, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            pruned_param = param.clone()
            # Create a mask: 1 where |value| >= threshold, 0 otherwise
            mask = pruned_param.abs() >= threshold
            # Zero out values below threshold
            pruned_param[~mask] = 0

            pruned_state_dict[key] = pruned_param
            pruning_mask_dict[key] = mask
        else:
            pruned_state_dict[key] = param

    return pruned_state_dict, pruning_mask_dict


if __name__ == "__main__":
    path_to_model = "LIG_S1/flash/task2/best_retrained_flashnet10.pt"
    nonzero_count, zero_count, size_bits, size_mb = measure_modelsize(path_to_model)
    print(f"Number of non-zero parameters: {nonzero_count}")
    print(f"Number of zero parameters: {zero_count}")
    print(f"Model size in bits: {size_bits:.2f}")
    print(f"Model size in MB: {size_mb:.2f}")
    # max_model_size_mb = 2.28
    # pruned_path = "LIG_S1/flash/task_common/last_flashnet10_pruned.pt"
    # mask_file = "LIG_S1/flash/task_common/last_flashnet10_mask.pkl"
    # pruned_dict, final_size, used_threshold = one_shot_prune_to_size(
    #     model_path=path_to_model,
    #     max_size_mb=max_model_size_mb,
    #     pruned_model_path=pruned_path,
    #     mask_path=mask_file,
    # )

    # print(f"Pruning threshold used: {used_threshold:.6g}")
    # print(f"Final pruned model size: {final_size:.4f} MB")
    # print(f"Pruned model saved to: {pruned_path}")
    # print(f"Pruning mask saved to: {mask_file}")
