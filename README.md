# Optimal Sample Selection Through Uncertainty Estimation and Its Application in Deep Learning(COPS) 

## Paper Link [arxiv](https://arxiv.org/abs/2309.02476)

# Requirements

## Environment

Our code works with the following environment:
* torch==2.0.1
* torchvision==0.15.2
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
* `dataset`: which dataset to use


The following parameters are used for uncertainty generation 
* `model_num`: number of models for uncertainty generation


The following parameters are used for sampling

* `test_mode` : sampling method , cops or random
* `sample_number` : sampling number per class
* `loss_type` : base for vanilla cross entropy, reweight clip for weighting cross entropy
* `constant_1` : clip threshold for sampling
* `constant_2` : clip threshold for reweighting


# Quick Start (For Reproducing Results)
## Step 1 Split Data into Probe Set and Sampling Set
### To run all the split at once
run the following code
```bash
sh split_data.sh
```
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
## Step 2 Run Uncertainty Generation
To generate the uncertainty for all the dataset
### To run all the split at once
```bash
sh get_uncertainty.sh
```

###  CIFAR10
run the following code
```bash
python uncertainty_generation.py --dataset cifar10 --lr 0.1 --model resnet20 --model_num 10 --num_classes 10 --epochs 100 --weight_folder cifar10_resnet20_uncertainty_weight --npy_folder cifar_npy
```

### For CIFAR10-N
run the following code
```bash
python uncertainty_generation.py --dataset cifar10_aggre --lr 0.1 --model resnet20 --model_num 10 --num_classes 10 --epochs cifar10_aggre --weight_folder cifar10_resnet20_uncertainty_weight --npy_folder cifar_aggre_npy
```


### For CIFAR100
run the following code
```bash
python uncertainty_generation.py --dataset cifar100 --lr 0.1 --model resnet20 --model_num 10 --num_classes 100 --epochs cifar10_aggre --weight_folder cifar100_resnet20_uncertainty_weight --npy_folder cifar100_npy
```

## Step 3 Run Retraining

### CIFAR10

sample number 300


To run the uniform basline, please run the following code, the result is 
#### Uniform Baseline  
```bash
python retraining_withy.py --seed 0 --dataset cifar --model resnet20 --test_mode random --sample_number 300 --loss_type base --epochs 150
```

To run the COPS with y, please run the following code, the result is 
#### COPS With Y

```bash
python retraining_withy.py --seed 0 --dataset cifar --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 300 --loss_type reweight_clip --epochs 150
```

To run the COPS without y, please run the following code, the result is 
#### COPS Without Y
```bash
python retraining_withy.py --seed 0 --dataset cifar --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 300 --loss_type reweight_clip --epochs 150
```


### CIFAR10-N

sample number 300



To run the uniform basline, please run the following code, the result is 
#### Uniform Baseline  
```bash
python retraining_withy.py --seed 0 --dataset cifar10_aggre --model resnet20 --test_mode random --sample_number 300 --loss_type base --epochs 150
```

To run the COPS with y, please run the following code, the result is 
#### COPS With Y
```bash
python retraining_withy.py --seed 0 --dataset cifar10_aggre --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 300 --loss_type reweight_clip --epochs 150
```
To run the COPS without y, please run the following code, the result is 
#### COPS Without Y
```bash
python retraining_withy.py --seed 0 --dataset cifar10_aggre --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 300 --loss_type reweight_clip --epochs 150
```

### CIFAR100

sample number 300



To run the uniform basline, please run the following code, the result is 
#### Uniform Baseline  
```bash
python retraining_withy.py --seed 0 --dataset cifar100 --model resnet20 --test_mode random --sample_number 30 --loss_type base --epochs 150
```

To run the COPS with y, please run the following code, the result is 
#### COPS With Y
```bash
python retraining_withy.py --seed 0 --dataset cifar100 --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 30 --loss_type reweight_clip --epochs 150
```
To run the COPS without y, please run the following code, the result is 
#### COPS Without Y
```bash
python retraining_withy.py --seed 0 --dataset cifar100 --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 30 --loss_type reweight_clip --epochs 150
```

# Contact Information

For help or issues using COPS, please submit a GitHub issue.
For personal communication related to COPS, please contact Yong Lin (`ylindf@connect.ust.hk`).

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