# cifar10
python uncertainty_generation.py --dataset cifar10 --lr 0.1 --model resnet20 --model_num 10 --num_classes 10 --epochs 100 --weight_folder cifar10_resnet20_uncertainty_weight --npy_folder cifar_npy

# cifar10-N
python uncertainty_generation.py --dataset cifar10_aggre --lr 0.1 --model resnet20 --model_num 10 --num_classes 10 --epochs 100 --weight_folder cifar10_resnet20_uncertainty_weight --npy_folder cifar_aggre_npy

# cifar100
python uncertainty_generation.py --dataset cifar100 --lr 0.1 --model resnet20 --model_num 10 --num_classes 100 --epochs 100 --weight_folder cifar100_resnet20_uncertainty_weight --npy_folder cifar100_npy