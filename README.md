# Mitigating Embedding and Class Assignment Mismatch in Unsupervised Image Classification #
This repository is the pytorch code for "Mitigating Embedding and Class Assignment Mismatch in Unsupervised Image Classification"
## Highlight ##
* Our two-stage process starts with embedding learning as a pretraining step, which produces a great initialization. The second stage then aims to assign a class for each data point by refining its pretrained embedding. Our model successfully optimizes two objectives without falling into the misaligned state.
* The proposed method outperforms the existing baselines substantially. With the CIFAR-10 dataset, we achieve an accuracy of 81.0%, whereas the best performing alternative reaches 61.7%.
* Extensive experiments and ablation studies confirm that both stages are critical to the overall performance gain. In-depth comparison with the current state-of-the-art (SOTA) methods reveals that a massive advantage of our approach comes from the embedding learning initialization that gathers similar images nearby even in a low-dimensional space.
* Our model can be adopted as a pretraining step for a semi-supervised task with few labels. We show the potential gain in the experiment section.

## Required packages ##
- python == 3.6.10
- pytorch == 1.1.0
- scikit-learn == 0.21.2
- scipy == 1.3.0
- numpy == 1.18.5
- pillow == 7.1.2


## Two stage model architecture ##
<center><img src="./fig/model_arch.png"> </center>

### (a) First stage : Unsupervised deep embedding
* * *
#### super_and.py 
The encoder projects input images to a lower dimension embedding sphere via deep embedding ([Super-AND](https://github.com/super-AND/super-AND)). The encoder is trained to gather samples with similar semantic contents nearby and separate them if otherwise.

```
usage: super_and.py [-h] [--dataset DATASET] [--low_dim LOW_DIM] [--npc_t T]
                    [--npc_m NPC_M] [--ANs_select_rate ANS_SELECT_RATE]
                    [--ANs_size ANS_SIZE] [--lr LR] [--momentum M]
                    [--weight_decay W] [--epochs EPOCHS] [--rounds ROUNDS]
                    [--batch_t T] [--batch_m N] [--batch_size B]
                    [--model_dir MODEL_DIR] [--resume RESUME] [--test_only]
                    [--seed SEED]
```
##### Example #####
```
python3 super_and.py --dataset cifar10
```




### (b) Second stage: Unsupervised class assignment with refining pretrained embeddings
* * *
### main.py
Multi-head normalized fully-connected layer classifies images by jointly optimizing the clustering and embedding losses.

```
usage: main.py [-h] [--dataset DATASET] [--low_dim LOW_DIM] [--lr LR]
               [--momentum M] [--weight_decay W] [--epochs EPOCHS]
               [--batch_t T] [--batch_m N] [--batch_size B]
               [--model_dir MODEL_DIR] [--resume RESUME] [--test_only]
               [--seed SEED]
```
##### Example #####

```
python3 main.py --dataset cifar10 --resume [first stage pretrained model]
```

<img src="./fig/stage2.png"> 

## Pretrained Model ##
Currently, we support the pretrained model for our model and super-AND on CIFAR10 dataset.
* [Our model](https://drive.google.com/file/d/1H3ppCkPQNHFEYQS4PLuV26Cp3HpbG4Nb/view?usp=sharing)
* [Super-AND](https://drive.google.com/file/d/1cABTquqOl5N2Wbchxs0-DBI6OVfnqY5J/view?usp=sharing)

## Result ##

### Unsupervised Image Classification Result ###
* We achieve new state of the art unsupervised image classification record on multiple dataset (CIFAR 10, CIFAR 100-20, STL 10)
<img src="./fig/model_result.png" width="500" height="400"> 



