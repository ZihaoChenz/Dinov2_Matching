import subprocess
import sys

train_configs = [
    {
        "ImageType": "JPG",
        "ResultFolder": "output/ablation_learning/cityscape_01/foggy19/neg_mul/foggy19_neg_mul_10",
        "DataFolder": "data/cityscape_01/foggy19",
        "json": "json/ablation_learning/cityscape_01/foggy19/neg_mul/foggy19_neg_mul_10"
    },
    {
        "ImageType": "JPG",
        "ResultFolder": "output/ablation_learning/cityscape_01/sunrise21/neg_mul/sunrise21_neg_mul_10",
        "DataFolder": "data/cityscape_01/sunrise21",
        "json": "json/ablation_learning/cityscape_01/sunrise21/neg_mul/sunrise21_neg_mul_10"
    },
    {
        "ImageType": "JPG",
        "ResultFolder": "output/ablation_learning/cityscape_01/snow/neg_mul/snow_neg_mul_10",
        "DataFolder": "data/cityscape_01/snow",
        "json": "json/ablation_learning/cityscape_01/snow/neg_mul/snow_neg_mul_10"
    },
    {
        "ImageType": "JPG",
        "ResultFolder": "output/ablation_learning/cityscape_01/autumn/neg_mul/autumn_neg_mul_10",
        "DataFolder": "data/cityscape_01/autumn",
        "json": "json/ablation_learning/cityscape_01/autumn/neg_mul/autumn_neg_mul_10"
    },
    {
        "ImageType": "JPG",
        "ResultFolder": "output/ablation_learning/cityscape_01/foggy19/neg_mul/foggy19_temperature_0.1",
        "DataFolder": "data/cityscape_01/foggy19",
        "json": "json/ablation_learning/cityscape_01/foggy19/neg_mul/foggy19_temperature_0.1"
    },
    {
        "ImageType": "JPG",
        "ResultFolder": "output/ablation_learning/cityscape_01/sunrise21/neg_mul/sunrise21_temperature_0.1",
        "DataFolder": "data/cityscape_01/sunrise21",
        "json": "json/ablation_learning/cityscape_01/sunrise21/neg_mul/sunrise21_temperature_0.1"
    },
    {
        "ImageType": "JPG",
        "ResultFolder": "output/ablation_learning/cityscape_01/snow/neg_mul/snow_temperature_0.1",
        "DataFolder": "data/cityscape_01/snow",
        "json": "json/ablation_learning/cityscape_01/sunrise21/neg_mul/snow_temperature_0.1"
    },
    {
        "ImageType": "JPG",
        "ResultFolder": "output/ablation_learning/cityscape_01/autumn/neg_mul/autumn_temperature_0.1",
        "DataFolder": "data/cityscape_01/autumn",
        "json": "json/ablation_learning/cityscape_01/autumn/neg_mul/autumn_temperature_0.1"
    },
    {
        "ImageType": "JPG",
        "ResultFolder": "output/ablation_learning/cityscape_01/autumn/neg_mul/autumn_temperature_0.5",
        "DataFolder": "data/cityscape_01/autumn",
        "json": "json/ablation_learning/cityscape_01/autumn/neg_mul/autumn_temperature_0.5"
    },
    {
        "ImageType": "JPG",
        "ResultFolder": "output/ablation_learning/cityscape_01/sunrise21/neg_mul/sunrise21_temperature_0.5",
        "DataFolder": "data/cityscape_01/sunrise21",
        "json": "json/ablation_learning/cityscape_01/sunrise21/neg_mul/sunrise21_temperature_0.5"
    },
    {
        "ImageType": "JPG",
        "ResultFolder": "output/ablation_learning/cityscape_01/foggy19/neg_mul/foggy19_temperature_0.5",
        "DataFolder": "data/cityscape_01/foggy19",
        "json": "json/ablation_learning/cityscape_01/foggy19/neg_mul/foggy19_temperature_0.5"
    },
    {
        "ImageType": "JPG",
        "ResultFolder": "output/ablation_learning/cityscape_01/snow/neg_mul/snow_temperature_0.5",
        "DataFolder": "data/cityscape_01/snow",
        "json": "json/ablation_learning/cityscape_01/snow/neg_mul/snow_temperature_0.5"
    }
]

for cfg in train_configs:
    cmd = [
        sys.executable,  # 使用当前Python解释器
        "visualize.py",
        "--ImageType", cfg["ImageType"],
        "--ResultFolder", cfg["ResultFolder"],
        "--DataFolder", cfg["DataFolder"],
        "--json", cfg["json"]
    ]

    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if process.returncode != 0:
        print(f"visualize fail: {cfg['json']}")
        print("错误信息:", process.stderr)
        break  # 可选择停止后续任务
    else:
        print(f"Finish: {cfg['json']}")