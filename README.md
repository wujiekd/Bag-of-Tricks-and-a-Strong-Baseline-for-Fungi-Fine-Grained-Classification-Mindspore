# Bag-of-Tricks-and-a-Strong-Baseline-for-Fungi-Fine-Grained-Classification

paper link: http://star.informatik.rwth-aachen.de/Publications/CEUR-WS/Vol-3180/paper-182.pdf

![image](https://user-images.githubusercontent.com/49955700/199949682-9eaaf8aa-2b84-4ce6-9cbd-04d42696d2cf.png)


This work is sponsored by Natural Science Foundation of China(62276242), CAAI-Huawei MindSpore Open Fund(CAAIXSJLJJ-2021-016B), Anhui Province Key Research and Development Program(202104a05020007), and USTC Research Funds of the Double First-Class Initiative(YD2350002001)”。


## 1. Environment setting 

### 1.0. Package
* Several important packages
    - mindspore == 1.8.1
    - mindcv
    
* Replace folder mindcv/ to our mindcv/ folder (We made some changes to the original Mindcv framework, such as adding TA, etc)  
    
    #### MindCV is an open source toolbox for computer vision research and development based on MindSpore. [mindcv](https://github.com/mindspore-ecosystem/mindcv)

### 1.1. Dataset
In this project, we use a large fungi's datasets from this challenge to evaluate performance:
* [Fungi2022](https://www.kaggle.com/competitions/fungiclef2022/data)

### 1.2. OS
- [x] Windows10
- [x] Ubuntu20.04
- [x] macOS (CPU only)

## 2. Train
- [x] Single GPU Training
- [x] DataParallel (single machine multi-gpus)
- [ ] DistributedDataParallel


### 2.1. data
train data and test data structure:  
```
├── DF20/
│   ├── img20001.jpg
│   ├── img20002.jpg
│   └── ....
├── DF21/
│   ├── img21001.jpg
│   ├── img21002.jpg
│   └── ....
└──
```
  
Training sets and test sets are distributed with CSV labels corresponding to them.

### 2.2. configuration
you can directly modify yaml file (in ./configs/)

### 2.3. run.
take the SwinTransformer training process as an example.

1.  we train a basic model by dividing the training set and verification set 9:1
```
python train.py ./Fungidata/DF20 -c configs/swin_large_384.yaml \
        --freeze-layer 2 \
        --batch-size 32 \
        --lr 0.01 \
        --decay-rate 0.9 \
        --output ./output/Swin-TF/DF20/freeze_layer_2
```

2. Add data augment and continue fine-tuning
```
python train.py ./Fungidata/DF20 -c configs/swin_large_384.yaml \
        --output ./output/Swin-TF/DF20/All_aug/freeze_layer_2 \
        --initial-checkpoint ./output/Swin-TF/DF20/freeze_layer_2/20220430-123449-swin_large_patch4_window12_384-384/Best_Top1-ACC.pth.tar \
        --freeze-layer 2 \
        --lr 0.001 \
        --batch-size 32 \
        --warmup-epochs 0 \
        --cutmix 1 \
        --color-jitter 0.4 \
        --reprob 0.25 \
        --aa trivial \
        --decay-rate 0.9
```

3. Modify the loss function and continue fine-tuning
```
python train.py ./Fungidata/DF20 -c configs/swin_large_384.yaml \
        --output ./output/Swin-TF/DF20/new_loss/freeze_layer_2 \
        --initial-checkpoint ./output/Swin-TF/DF20/All_aug/freeze_layer_2/20220430-123449-swin_large_patch4_window12_384-384/Best_Top1-ACC.pth.tar \
        --freeze-layer 2 \
        --lr 0.001 \
        --batch-size 32 \
        --warmup-epochs 0 \
        --cutmix 1 \
        --color-jitter 0.4 \
        --reprob 0.25 \
        --aa trivial \
        --decay-rate 0.9 \
        --Focalloss
```

4. Fine-tuning with full data sets
```
python train_all.py ./Fungidata/DF20 -c configs/swin_large_384.yaml \
         --batch-size 32 \
         --img-size 384 \
         --output ./output/Swin-TF/DF20/All_data/swin_large_384 \
         --freeze-layer 2 \
         --initial-checkpoint ./output/Swin-TF/DF20/new_loss/freeze_layer_2/20220502-114033-swin_large_patch4_window12_384-384/Best_Top1-ACC.pth.tar \
         --lr 0.001 \
         --cutmix 1 \
         --color-jitter 0.4 \
         --reprob 0.25 \
         --aa trivial \
         --decay-rate 0.1 \
         --warmup-epochs 0 \
         --epochs 24 \
         --sched multistep \
         --checkpoint-hist 24 \
         --Focalloss
```

5. Two-stage training
```
python train_all.py ./Fungidata/DF20 -c configs/swin_large_384.yaml \
         --batch-size 32 \
         --img-size 384 \
         --output ./output/Swin-TF/DF20/two_stage/swin_large_384 \
         --freeze-layer 2 \
         --initial-checkpoint ./output/Swin-TF/DF20/All_data/freeze_layer_2/20220504-115867-swin_large_patch4_window12_384-384/Best_Top1-ACC.pth.tar \
         --lr 0.001 \
         --cutmix 1 \
         --color-jitter 0.4 \
         --reprob 0.25 \
         --aa trivial \
         --decay-rate 0.1 \
         --warmup-epochs 0 \
         --epochs 5 \
         --sched multistep \
         --Focalloss
```

### 2.4. multi-gpus
change train.sh
```
python -m mindspre.distributed.launch --nproc_per_node=4 train_all.py
```  

## 3. Evaluation
for details, see test.sh
```
sh test.sh
```

Use the sh test.sh command to execute the reasoning file (the checkpoint in the google cloud disk will be used, please download it in 5. Challenge's final checkpoints and logits)

## 4. Model Ensemble
run model_ensemble.ipynb

Execute 3. Evaluation to get the logits file required by model_ensemble.ipynb (you can also download it directly from google cloud disk, see 5. Challenge's final checkpoints and logits for details)

## 5. Challenge's final checkpoints and logits
It can be downloaded from Google Cloud Disk: https://drive.google.com/file/d/1v9SlsjXXKI5kizg9BTMWfA9faaawAgpv/view?usp=sharing

  
It can be directly used for model ensemble reasoning.

### Acknowledgment

* Thanks to [mindcv](https://github.com/mindspore-ecosystem/mindcv) for Mindspore implementation.
