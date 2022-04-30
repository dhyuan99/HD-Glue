for size in 100 500 1000 5000
do
    for dim in 2000 4000 8000
    do
        python train_HD.py -size $size -dim $dim -dataset CIFAR10
    done
done

for size in 100 500 1000 5000
do
    for dim in 2000 4000 8000
    do
        python train_HD.py -size $size -dim $dim -dataset CIFAR100
    done
done
