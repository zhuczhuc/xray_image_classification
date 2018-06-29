## 使用方法：
1. 原始数据的不同分类放在 data/original/下，每类图像单独放在一个目录下，目录的名称最好是利用图像分类的名称，运行以下命令，预处理数据：
```bash
python build_dataset.py --data_dir data/original --output_dir data/64x64_data
```