for dataset in 'CIFAR100' 'CIFAR10'
do
    for net in 'vgg11' 'vgg13' 'vgg16' 'vgg19' 'resnet18' 'resnet34'
    do
        python train_enc.py -dataset $dataset -net $net -gpu
    done
done
