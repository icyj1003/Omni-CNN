import os
import subprocess
import sys
import argparse

argument_parser = argparse.ArgumentParser(description="Run experiments for Omni-CNN")
argument_parser.add_argument(
    "--run-heterogeneous",
    action="store_true",
    help="Whether to run heterogeneous experiments (default: False)",
)
argument_parser.add_argument(
    "--run-scaling",
    action="store_true",
    help="Whether to run scaling experiments (default: False)",
)
args = argument_parser.parse_args()

gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
exp = "LIG_S1"
# CUDA_VISIBLE_DEVICES=$2 python train.py --dataset flash --exp_name flash --base_path flash_GPS_Image_LiDAR --save_path experiments/$1/ --load-cummu-model experiments/$1/flash/task2/cumu_model.pt --classes 64 --mixup --alpha 0 --smooth --smooth-eps 0.1 --config-setting 3,5,2 --arch flashnet --depth 10 --tasks 3 --learning-mode federated --use_tfed\

base_cmd = [
    sys.executable,
    "train.py",
    "--dataset",
    "flash",
    "--exp_name",
    "flash",
    "--base_path",
    "flash_GPS_Image_LiDAR",
    "--save_path",
    f"experiments/LIG_S1/",
    "--load-cummu-model",
    "experiments/LIG_S1/flash/task2/cumu_model.pt",
    "--classes",
    "64",
    "--mixup",
    "--alpha",
    "0",
    "--smooth",
    "--smooth-eps",
    "0.1",
    "--config-setting",
    "3,5,2",
    "--arch",
    "flashnet",
    "--depth",
    "10",
    "--tasks",
    "3",
    "--learning-mode",
    "federated",
    "--lr",
    "0.001",
]

client_settings = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 3, 4, 5, 7],
    [0, 3, 4, 7],
    [0, 3],
]

# scaling experiments
for clients in client_settings:
    for use_tfed in [True, False]:
        cmd = base_cmd + ["--clients", *map(str, clients)]
        if use_tfed:
            cmd.append("--use_tfed")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

# For heterogeneous experiments, you can modify the client_settings and the base_cmd accordingly.
heterogeneous_settings = [0, 1, 2, 3]
if args.run_heterogeneous:
    for use_tfed in [True, False]:
        for heto in heterogeneous_settings:
            cmd = base_cmd + ["--heterogeneous", str(heto)]
            # cmd = cmd + [
            #     "--comms-round",
            #     "5",
            # ]  # You can adjust the number of communication rounds as needed
            cmd = cmd + ["--remove_size_limit"]
            if use_tfed:
                cmd.append("--use_tfed")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True, env=env)
