import torch
import sys
import numpy as np

class EntireDataset(torch.utils.data.Dataset):
    def __init__(self, train_X, train_Y, test_X, test_Y, train=True):

        if train:
            self.X = train_X
            self.Y = train_Y
        else:
            self.X = test_X
            self.Y = test_Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_network(args):

    n_classes = int(args.dataset.strip('CIFAR'))

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn(n_classes)
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn(n_classes)
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn(n_classes)
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn(n_classes)
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(n_classes)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(n_classes)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu:
        net = net.cuda()

    return net
