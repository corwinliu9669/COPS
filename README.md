# Optimal Sample Selection Through Uncertainty Estimation and Its Application in Deep Learning 
## Introduction
We explore sub-sampling techniques on large datasets for training deep models. We propose an optimal sampling strategy, COPS (unCertainty based OPtimal Sub-sampling), that minimizes expected generalization loss for coreset selection and active learning in linear softmax regression. Unlike existing methods, our approach avoids the need for computing the inverse of the covariance matrix by leveraging neural network outputs. We also address the proposed method's sensitivity to model mis-specification in low-density regions through empirical analysis. Our methods consistently outperform baselines across diverse datasets and architectures, demonstrating their superior performance and effectiveness. Paper Link is here on [arxiv](https://arxiv.org/abs/2309.02476)

# Requirements

## Environment

Our code works with the following environment:
* torch==2.0.1
* torchvision==0.15.2
* torchtext==0.15.2
* PIL==9.4.0

To install the necessary packages for the project, please run: 
```
pip install -r requirements.txt
```
## Datasets
Run the following command to download the datasets:
```bash
sh data_downloader.sh
```

For CIFAR10-N, download the label from this [link](https://github.com/UCSC-REAL/cifar-10-100n/blob/main/data/CIFAR-10_human.pt)


Here we can download CIFAR10, CIFAR10-N, CIFAR100
# Parameters
The following parameters are used for training 
* `lr`: learning rate
* `epochs`: training epochs
* `dataset`: which dataset to use, `cifar' for CIFAR10, `cifar10_aggre' for CIFAR10-N, `cifar100' for CIFAR100, `imdb' for IMDB


The following parameters are used for uncertainty generation 
* `model_num`: number of models for uncertainty generation


The following parameters are used for sampling

* `test_mode` : sampling method, oracle_sampling_cut for cops or random for uniform
* `sample_number` : sampling number per class
* `loss_type` : base for vanilla cross entropy, reweight clip for weighting cross entropy
* `constant_1` : clip threshold for sampling
* `constant_2` : clip threshold for reweighting


# Quick Start (For Reproducing Results)
## Step 1 Split Data into Probe Set and Sampling Set

###  CIFAR10
run the following code
```bash
python split_data/cifar_split.py
```

### For CIFAR10-N
run the following code
```bash
python split_data/cifarn_split.py
```

### For CIFAR100
run the following code
```bash
python split_data/cifar100_split.py
```
### For IMDB 
the split code is combined with training code
## Step 2 Run Uncertainty Generation

### To run uncertainty for all datasets including (CIFAR10...) at once
```bash
sh get_uncertainty.sh
```

###  To run uncertainty for CIFAR10
run the following code
```bash
python uncertainty_generation.py --dataset cifar10 --lr 0.1 --model resnet20 --model_num 10 --num_classes 10 --epochs 100 --weight_folder cifar10_resnet20_uncertainty_weight --npy_folder cifar_npy
```

### For CIFAR10-N
run the following code
```bash
python uncertainty_generation.py --dataset cifar10_aggre --lr 0.1 --model resnet20 --model_num 10 --num_classes 10 --epochs 100 --weight_folder cifar10_resnet20_uncertainty_weight --npy_folder cifar_aggre_npy
```


### For CIFAR100
run the following code
```bash
python uncertainty_generation.py --dataset cifar100 --lr 0.1 --model resnet20 --model_num 10 --num_classes 100 --epochs 100  --weight_folder cifar100_resnet20_uncertainty_weight --npy_folder cifar100_npy
```

### For IMDB
run the following code
```bash
python uncertainty_generation_imdb.py --lr 1e-3 --model gru --model_num 10  --epochs 20  --weight_folder imdb_uncertainty_weight --npy_folder imdb_npy
```
## Step 3 Run Sampling and Training
### To run sampling and training for all datasets including (CIFAR10...) at once

To get the results of three datasets, please run the following code
```bash
sh retraining.sh
```
### CIFAR10

sample number 300


To run the uniform basline, please run the following code, the result is 57.97
#### Uniform Baseline  
```bash
python retraining_withy.py  --seed 1 --lr 1e-3 --dataset cifar --model resnet20 --test_mode random --sample_number 300 --loss_type base --epochs 150
```

To run the COPS with y, please run the following code, the result is 61.82
#### COPS With Y

```bash
python retraining_withy.py --seed 1 --lr 1e-3 --dataset cifar --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 300 --loss_type reweight_clip --epochs 150
```

To run the COPS without y, please run the following code, the result is 61.22
#### COPS Without Y
```bash
python retraining_woy.py --seed 1 --lr 1e-3 --dataset cifar --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 300 --loss_type reweight_clip --epochs 150
```


### CIFAR10-N

sample number 300



To run the uniform basline, please run the following code, the result is 57.49
#### Uniform Baseline  
```bash
python retraining_withy.py --seed 1 --lr 1e-3 --dataset cifar10_aggre --model resnet20 --test_mode random --sample_number 300 --loss_type base --epochs 150
```

To run the COPS with y, please run the following code, the result is 60.73
#### COPS With Y
```bash
python retraining_withy.py --seed 1 --lr 1e-3 --dataset cifar10_aggre --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 300 --loss_type reweight_clip --epochs 150
```
To run the COPS without y, please run the following code, the result is 58.09
#### COPS Without Y
```bash
python retraining_woy.py --seed 1 --lr 1e-3 --dataset cifar10_aggre --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 300 --loss_type reweight_clip --epochs 150
```

### CIFAR100

sample number 300

To run the uniform basline, please run the following code, the result is 19.01
#### Uniform Baseline  
```bash
python retraining_withy.py --seed 1 --lr 1e-3 --dataset cifar100 --model resnet20 --test_mode random --sample_number 30 --loss_type base --epochs 150
```

To run the COPS with y, please run the following code, the result is 20.81

#### COPS With Y
```bash
python retraining_withy.py --seed 1 --lr 1e-3 --dataset cifar100 --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 30 --loss_type reweight_clip --epochs 150
```
To run the COPS without y, please run the following code, the result is 22.49
#### COPS Without Y
```bash
python retraining_woy.py --seed 1 --lr 1e-3 --dataset cifar100 --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 30 --loss_type reweight_clip --epochs 150
```


### IMDB

sample number 1000

To run the uniform basline, please run the following code, the result is 77.61
#### Uniform Baseline  
```bash
python retraining_imdb.py --seed 1 --lr 1e-3 --dataset imdb --model gru --test_mode random --sample_number 1000 --loss_type base --epochs 20
```

To run the COPS with y, please run the following code, the result is 78.49

#### COPS With Y
```bash
python retraining_imdb.py --seed 1 --lr 1e-3 --un_type wy --dataset imdb --model gru --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 1000 --loss_type reweight_clip --epochs 20
```
To run the COPS without y, please run the following code, the result is 79.87
#### COPS Without Y
```bash
python retraining_imdb.py --seed 1 --lr 1e-3 --un_type woy --dataset imdb --model gru --test_mode oracle_sampling_cut_whole --constant_1 0.005 --constant_2 10 --sample_number 1000 --loss_type reweight_clip --epochs 20
```


# Contact Information

For help or issues using COPS, please submit a GitHub issue.


# Reference 
If you use or extend our work, please cite the following paper:
```
@article{lin2023optimal,
  title={Optimal Sample Selection Through Uncertainty Estimation and Its Application in Deep Learning},
  author={Lin, Yong and Liu, Chen and Ye, Chenlu and Lian, Qing and Yao, Yuan and Zhang, Tong},
  journal={arXiv preprint arXiv:2309.02476},
  year={2023}
}
```
