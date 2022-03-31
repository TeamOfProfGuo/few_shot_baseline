# Few-Shot Meta-Baseline

Adopted from <a href="https://github.com/yinboc/few-shot-meta-baseline" target="_blank">Meta-Baseline</a>.

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

Download via <a href="https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing" target="_blank">this link</a> (credits <a href="https://github.com/gidariss/FewShotWithoutForgetting" target="_blank">Spyros Gidaris</a>), and transfer the zip file to your dataset folder on Greene.
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

## Training
**Note:** Modify the path in slurm scripts (as needed) before you start.
```
# switch to project root
cd /scratch/$USER/few_shot_baseline

# train classifier
sbatch train_classifier.slurm mini

# After the job ends:
cd ./log/mini && ls
cat train_classifier_mini_[some_time_info].log
# [To be updated]
```
