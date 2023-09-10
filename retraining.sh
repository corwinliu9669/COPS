# cifar10
### n sample 300
mkdir -p retrain_logs
##### random
python retraining_withy.py --seed 1 --lr 1e-3 --dataset cifar --model resnet20 --test_mode random --sample_number 300 --loss_type base  --epochs 150 --un_folder cifar_npy > retrain_logs/cifar_random.log
##### cops wy
python retraining_withy.py --seed 1 --lr 1e-3 --dataset cifar --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 300 --loss_type reweight_clip  --epochs 150 --un_folder cifar_npy > retrain_logs/cifar_cops_wy.log
##### cops woy
python retraining_woy.py --seed 1 --lr 1e-3 --dataset cifar --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 300 --loss_type reweight_clip  --epochs 150 --un_folder cifar_npy  >  retrain_logs/cifar_cops_woy.log

# cifar10-N
### n sample 300
##### random
python retraining_withy.py --seed 1 --lr 1e-3 --dataset cifar10_aggre --model resnet20 --test_mode random --sample_number 300 --loss_type base  --epochs 150 --un_folder cifar_noise_npy > retrain_logs/cifar_aggre_random.log
##### cops wy
python retraining_withy.py --seed 1 --lr 1e-3 --dataset cifar10_aggre --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 300 --loss_type reweight_clip  --epochs 150 --un_folder cifar_noise_npy > retrain_logs/cifar_aggre_cops_wy.log
##### cops woy
python retraining_woy.py --seed 1 --lr 1e-3 --dataset cifar10_aggre --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 300 --loss_type reweight_clip  --epochs 150 --un_folder cifar_noise_npy > retrain_logs/cifar_aggre_cops_woy.log


# cifar100
### n sample 30
##### random
python retraining_withy.py --seed 1 --lr 1e-3 --dataset cifar100 --model resnet20 --test_mode random --sample_number 30 --loss_type base --epochs 150 --un_folder cifar100_npy > retrain_logs/cifar100_random.log
##### cops wy
python retraining_withy.py --seed 1 --lr 1e-3 --dataset cifar100 --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 30 --loss_type reweight_clip  --epochs 150 --un_folder cifar100_npy > retrain_logs/cifar100_cops_wy.log
##### cops woy
python retraining_woy.py --seed 1 --lr 1e-3 --dataset cifar100 --model resnet20 --test_mode oracle_sampling_cut --constant_1 0.005 --constant_2 10 --sample_number 30 --loss_type reweight_clip  --epochs 150 --un_folder cifar100_npy > retrain_logs/cifar100_cops_woy.log

