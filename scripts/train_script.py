import os
import subprocess
import sys

# train_list = [
#     "python train.py --DataFolder datasets/cityscape_stuttgart_00 --CheckPoints checkpoints/cityscape_00/ablation_learning/neg_mul_2 --batch_size 32 --epoch 150 --config config/neg_mul/config_2.yaml",
#     "python train.py --DataFolder datasets/cityscape_stuttgart_00 --CheckPoints checkpoints/cityscape_00/ablation_learning/neg_mul_20 --batch_size 4 --epoch 150 --config config/neg_mul/config_20.yaml",
#     "python train.py --DataFolder datasets/cityscape_stuttgart_00 --CheckPoints checkpoints/cityscape_00/ablation_learning/block_11_10_9_8 --batch_size 16 --epoch 150 --config config/block/config_11_10_9_8.yaml",
#     "python train.py --DataFolder datasets/cityscape_stuttgart_00 --CheckPoints checkpoints/cityscape_00/ablation_learning/temperature_0.1 --batch_size 16 --epoch 150 --config config/temperature/config_0.1.yaml",
#     "python train.py --DataFolder datasets/cityscape_stuttgart_00 --CheckPoints checkpoints/cityscape_00/ablation_learning/temperature_0.02 --batch_size 16 --epoch 150 --config config/temperature/config_0.02.yaml"
# ]
#
# for i in train_list:
#     os.system(i)



train_configs = [
    {
        "DataFolder": "datasets/cityscape_stuttgart_00",
        "CheckPoints": "checkpoints/cityscape_00/ablation_learning/neg_mul_100",
        "batch_size": 2,
        "epoch": 150,
        "config": "config/neg_mul/config_100.yaml"
    }
]

for cfg in train_configs:
    cmd = [
        sys.executable,  # 使用当前Python解释器
        "train.py",
        "--DataFolder", cfg["DataFolder"],
        "--CheckPoints", cfg["CheckPoints"],
        "--batch_size", str(cfg["batch_size"]),
        "--epoch", str(cfg["epoch"]),
        "--config", cfg["config"]
    ]

    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if process.returncode != 0:
        print(f"训练失败: {cfg['CheckPoints']}")
        print("错误信息:", process.stderr)
        break  # 可选择停止后续任务
    else:
        print(f"训练完成: {cfg['CheckPoints']}")