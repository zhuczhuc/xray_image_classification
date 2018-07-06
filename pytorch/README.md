## 使用方法：
1. 预处理数据：原始数据的不同分类放在 data/original/下，每类图像单独放在一个目录下，目录的名称最好是利用图像分类的名称，运行以下命令，预处理数据：

```bash
python build_dataset.py --data_dir data/original --output_dir data/64x64_data
```

2. 准备训练数据：创建experiments目录，并在其目录下创建base_model子目录，在base_model中再创建params.json文件，存放训练配置参数，内容如下：
```json
{
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 30,
    "dropout_rate":0.8,
    "num_channels": 32,
    "save_summary_steps": 100,
    "num_workers": 4
}
```
3. 训练数据：运行下面命令
```
python train.py --data_dir data/64x64_SIGNS --model_dir experiments/base_model
```
4. 预测单幅图像：
```
python --datafile data/original/visible/img.png
```