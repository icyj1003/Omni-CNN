import os
import subprocess
import sys

gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
exp = "LIG_S1"

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
    "--epochs",
    "300",
    "--mixup",
    "--alpha",
    "0",
    "--smooth",
    "--smooth-eps",
    "0.1",
    "--seed",
    "1",
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
    "--batch-size",
    "64",
    "--momentum",
    "0.9",
    "--weight-decay",
    "0.0001",
    "--comms-round",
    "100",
]

client_settings = [list(range(n)) for n in range(10, 1, -2)]

for use_tfed in [True]:
    for clients in client_settings:
        cmd = base_cmd + ["--clients", *map(str, clients)]
        if use_tfed:
            cmd.append("--use_tfed")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)
