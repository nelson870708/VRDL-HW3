# VRDL HW3
This is homework 3 in NCTU Selected Topics in Visual Recognition using Deep Learning.

## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04 LTS
- Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz
- NVIDIA RTX 2070

## Installation
1. Using Anaconda is strongly recommended. {envs_name} is the new environment name which you should assign.
```shell
conda create -n {envs_name} python=3.7
conda activate {envs_name}
```
2. Follow [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install detectron2.

3. Use the following command to install other requirements.
```shell
pip install -r requirement.txt
```

## Dataset Preparation
You can download the data [here](https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK). The label is in pascal format and is a json file, called "pascal_train.json".

### Prepare Data and Code
After downloading and extracting, the data directory is structured as:
```text
+- data
  +- train_images
    +- 2007_000033.jpg
    +- 2007_000042.jpg
    ...
  +- test_images
    +- 2007_000629.jpg
    +- 2007_001175.jpg
    ...
  pascal_train.json
  test.json
train.py
utils.py
cocosplit.py
make_submission.py
```

### Data Preprocessing
The following command is going to split the training data randomly by marking training data and validation data in two json files, called "train.json" and "val.json", respectively.
The ratio of the training data and validation data is 8 : 2.
```shell
python3 cocosplit.py --having-annotations -s 0.8 ./data/pascal_train.json ./data/train.json ./data/val.json
```
You can also using the following command for help.
```shell
python3 cocosplit.py -h
```
The code is modified from [here](https://github.com/akarazniewicz/cocosplit).

## Training
The code not only trains, but also valid the model.
You can train the model by following:
```shell
python3 train.py
```

## Testing
```shell
python3 make_submission.py
```