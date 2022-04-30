# HD-Glue

The HDC utilities is implemented using Cython. First run `python setup.py build_ext --inplace` to compile the cython code.

Download the MNIST, CIFAR10, CIFAR100 data from [here](https://drive.google.com/drive/folders/14_irBxvDdGZAcpEYJ7sjfQhuZI-uJ3Me?usp=sharing). Place the data in `HD-Glue/data`.

### Full Classifier Experiment

We first train the end-to-end classifier and use the classifier as the image encoder. For example: `python train_enc.py -net vgg11 -dataset CIFAR10 -gpu`. This experiment only supports `CIFAR10` and `CIFAR100`. The network can be chosen from `vgg11`, `vgg13`, `vgg16`, `vgg19`, `resnet18`, `resnet34`. To all possible combinations, please refer to `train_enc.sh`.

The simplest way to train is `sh train_enc.sh`. If you don't have a gpu, you should remove the `-gpu` argument.

If you don't want to train the network by yourself, you could download the pretrained models [here](https://drive.google.com/drive/folders/1mHG6_CDXlacwuU5DT_k7-tyxWcyfSpKU?usp=sharing) and place them to `HD-Glue/networks`.

After obtaining the neural networks, run `python gen_emb.py -dataset CIFAR10` and `python gen_emb.py -dataset CIFAR100` to obtain the embeddings given by each model. The embeddings will be saved to `HD-Glue/emb/`.

If you don't want to compute the embeddings by yourself, you can download the pre-computed embeddings [here](https://drive.google.com/drive/folders/10sl3PKY4TkNlnvvwZbXIExS2ZepDcfN1?usp=sharing) and place them to `HD-Glue/emb/`.

After saving the embeddings, run e.g. `python train_HD.py -size 1000 -dim 8000 -dataset CIFAR10` to get the experiment results. To see all the combinations, you can check and run `sh train_HD.sh`. 

