mkdir -p ./raw_data
cd ./raw_data
## cifar10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz 
tar -xvf cifar-10-python.tar.gz
## cifar100
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz 
tar -xvf cifar-100-python.tar.gz
## IMDB
## via torchtext
