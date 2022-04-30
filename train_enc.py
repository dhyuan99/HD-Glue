import numpy as np
import torch
from utils import EntireDataset, get_network
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-dataset', type=str, required=True, 
                        help='the dataset you want to work on')
    parser.add_argument('-gpu', action='store_true', default=False, 
                        help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    args = parser.parse_args()

    if args.dataset == 'MNIST':
        raise Exception("MNIST is not supported for this experiment.")

    train_X = np.load(f'data/{args.dataset}/train_X.npy')
    train_Y = np.load(f'data/{args.dataset}/train_Y.npy')
    test_X = np.load(f'data/{args.dataset}/test_X.npy')
    test_Y = np.load(f'data/{args.dataset}/test_Y.npy')

    transform_train = torch.nn.Sequential(
        transforms.RandomCrop(32, padding=[4]),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    )

    transform_test = torch.nn.Sequential(
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    )

    transform_train = torch.jit.script(transform_train)
    transform_test = torch.jit.script(transform_test)

    trainset = EntireDataset(train_X, train_Y, test_X, test_Y, train=True)
    train_loader = DataLoader(
            trainset, shuffle=True, num_workers=4, batch_size=args.b)
    
    testset = EntireDataset(train_X, train_Y, test_X, test_Y, train=False)
    test_loader = DataLoader(
            testset, shuffle=False, num_workers=4, batch_size=args.b)

    loss_fn = torch.nn.CrossEntropyLoss()

    net = get_network(args)
    net.train()
    optimizer = torch.optim.SGD(
            net.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for it in range(200):
        for i, (img, label) in enumerate(train_loader):
            img = transform_train(img)
            if args.gpu:
                img, label = img.cuda(), label.cuda()
            optimizer.zero_grad()
            yhat = net(img)
            loss = loss_fn(yhat, label)
            loss.backward()
            optimizer.step()

        acc, count = 0, 0
        for img, label in test_loader:
            img = transform_test(img)
            if args.gpu:
                img, label = img.cuda(), label.cuda()
            yhat = net(img).squeeze()
            acc += torch.sum(torch.argmax(yhat, axis=1) == label)
            count += len(yhat)

        print(f"Iter {it}: {round((acc/count).item()*100, 1)}%.")

        torch.save(net.state_dict(), f'networks/{args.dataset}/{args.net}.pth')

        scheduler.step()

