# Few-Shot Meta-Baseline

Adopted from [Few-Shot Meta-Baseline](https://github.com/yinboc/few-shot-meta-baseline).

---

## Requisites
- Test Env: Python 3.9.7 (Singularity)
- Packages:
    - torch (1.10.2+cu113), torchvision (0.11.3+cu113)
    - numpy, tensorboardX, tqdm

---

## Clone codebase
```
cd /scratch/$USER
git clone https://github.com/TeamOfProfGuo/few_shot_baseline -b hmd-base
cd few_shot_baseline
```

---

## Prepare mini-ImageNet dataset

Download via [**this link**](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) (credits [Spyros Gidaris](https://github.com/gidariss/FewShotWithoutForgetting)), and transfer the zip file to your dataset folder on Greene.
```
# switch to your dataset folder (path can be different)
cd /scratch/$USER/dataset

# create dataset folder & unzip
mkdir mini-imagenet
mv miniImageNet.zip ./mini-imagenet
cd mini-imagenet
unzip miniImageNet.zip

# create a soft link to this dir
# note: always use absolute path here
ln -s /scratch/$USER/dataset/mini-imagenet /scratch/$USER/few_shot_baseline/materials/mini-imagenet
```

---

## Training

### Train Classifier
**Note:** Modify the path in slurm scripts (as needed) before you start.
```
# switch to project root
cd /scratch/$USER/few_shot_baseline

# train classifier (on mini-ImageNet)
sbatch train_classifier.slurm mini
# => correspond to ./configs/train_classifier_mini.yaml

# after the job starts:
head ./save/classifier_mini-imagenet_resnet12/log.txt
# train dataset: torch.Size([3, 80, 80]) (x38400), 64
# val dataset: torch.Size([3, 80, 80]) (x18748), 64
# fs dataset: torch.Size([3, 80, 80]) (x12000), 20
# num params: 8.0M
# epoch 1, train 3.6071|0.1228, val 3.3443|0.1664, 1.0m 1.0m/1.7h
# [...]

# after the job ends:
tail ./save/classifier_mini-imagenet_resnet12/log.txt
# [...]
# epoch 98, train 0.7131|0.8050, val 0.7739|0.7898, 58.7s 1.9h/2.0h
# epoch 99, train 0.6956|0.8072, val 0.7852|0.7884, 58.8s 1.9h/1.9h
# epoch 100, train 0.6757|0.8103, val 0.7822|0.7872, fs 1: 0.6026 5: 0.7813, 1.9m 2.0h/2.0h
```

### Train Meta
**Note:** Again, modify the path in slurm scripts (as needed) before you start.
```
# switch to project root
cd /scratch/$USER/few_shot_baseline

# train meta (on mini-ImageNet)
sbatch train_meta.slurm mini
# => correspond to ./configs/train_meta_mini.yaml

# after the job starts:
head ./save/meta_mini-imagenet-1shot_meta-baseline-resnet12/log.txt
# train dataset: torch.Size([3, 80, 80]) (x38400), 64 classes
# tval dataset: torch.Size([3, 80, 80]) (x12000), 20 classes
# val dataset: torch.Size([3, 80, 80]) (x9600), 16 classes
# num params: 8.0M
# epoch 1, train 0.5586|0.8267, tval 0.9786|0.6146, val 0.9731|0.6224, 1.7m 1.7m/34.1m (@7)
# [...]

# after the job ends:
tail ./save/meta_mini-imagenet-1shot_meta-baseline-resnet12/log.txt
# [...]
# epoch 18, train 0.2906|0.9193, tval 0.9387|0.6247, val 0.9061|0.6481, 1.7m 30.4m/33.8m (@7)
# epoch 19, train 0.2923|0.9173, tval 0.9465|0.6210, val 0.9042|0.6478, 1.7m 32.1m/33.8m (@7)
# epoch 20, train 0.2755|0.9238, tval 0.9467|0.6213, val 0.9007|0.6501, 1.7m 33.8m/33.8m (@7)
```

### Test
**Note:** Again, modify the path in slurm scripts (as needed) before you start.
```
# switch to project root
cd /scratch/$USER/few_shot_baseline

# test (1-shot case on mini-ImageNet)
sbatch test_few_shot.slurm 1 mini
# => correspond to ./configs/test_few_shot_mini.yaml

# after the job ends:
cat ./save/test_mini-imagenet-1shot_meta-baseline-resnet12/log.txt
# dataset: torch.Size([3, 80, 80]) (x12000), 20
# num params: 8.0M
# test epoch 1: acc=62.55 +- 0.71 (%), loss=0.9358 (@7)
# test epoch 2: acc=62.48 +- 0.53 (%), loss=0.9344 (@2)
# test epoch 3: acc=62.54 +- 0.43 (%), loss=0.9339 (@0)
# test epoch 4: acc=62.61 +- 0.37 (%), loss=0.9329 (@18)
# test epoch 5: acc=62.79 +- 0.33 (%), loss=0.9292 (@6)
# test epoch 6: acc=62.85 +- 0.30 (%), loss=0.9276 (@16)
# test epoch 7: acc=62.85 +- 0.28 (%), loss=0.9274 (@13)
# test epoch 8: acc=62.90 +- 0.26 (%), loss=0.9258 (@0)
# test epoch 9: acc=62.81 +- 0.25 (%), loss=0.9272 (@10)
# test epoch 10: acc=62.82 +- 0.23 (%), loss=0.9271 (@9)
```