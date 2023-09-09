# cifar10
### n sample 300

##### random
python retraining_withy.py --seed 0 --dataset cifar --model resnet20 --test_mode random --sample_number 300 --loss_type base  --epochs 150 --un_folder cifar_npy
##### cops wy
python retraining_withy.py --seed 0 --dataset cifar --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 300 --loss_type reweight_clip  --epochs 150 --un_folder cifar_npy
##### cops woy
python retraining_withy.py --seed 0 --dataset cifar --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 300 --loss_type reweight_clip  --epochs 150 --un_folder cifar_npy

# cifar10-N
### n sample 300
##### random
python retraining_withy.py --seed 0 --dataset cifar10_aggre --model resnet20 --test_mode random --sample_number 300 --loss_type base  --epochs 150 --un_folder cifar_noise_npy
##### cops wy
python retraining_withy.py --seed 0 --dataset cifar10_aggre --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 300 --loss_type reweight_clip  --epochs 150 --un_folder cifar_noise_npy
##### cops woy
python retraining_withy.py --seed 0 --dataset cifar10_aggre --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 300 --loss_type reweight_clip  --epochs 150 --un_folder cifar_noise_npy


# cifar100
### n sample 30
##### random
python retraining_withy.py --seed 0 --dataset cifar100 --model resnet20 --test_mode random --sample_number 30 --loss_type base --epochs 150 --un_folder cifar100_npy
##### cops wy
python retraining_withy.py --seed 0 --dataset cifar100 --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 30 --loss_type reweight_clip  --epochs 150 --un_folder cifar100_npy
##### cops woy
python retraining_withy.py --seed 0 --dataset cifar100 --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 30 --loss_type reweight_clip  --epochs 150 --un_folder cifar100_npy

