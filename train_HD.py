import numpy as np
from Trainer import HDTrainer, classical_classification
from hdc import Aggregator, RecordBased

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-size', type=int, help='size of the training set')
parser.add_argument('-dim', type=int, help='HD Dimension')
parser.add_argument('-dataset', type=str, help='number of models')
args = parser.parse_args()

n_classes = int(args.dataset.strip('CIFAR'))

def get_uniform_trainset(train_X, train_Y, size):
    n_classes = len(np.unique(train_Y))
    labels = [[] for _ in range(n_classes)]
    for i in range(len(train_X)):
        labels[train_Y[i]].append(i)
    indices = []
    for i in range(size):
        indices.append(labels[i % n_classes][i // n_classes])
    return train_X[indices], train_Y[indices]

train_real_X = np.load(f'emb/{args.dataset}/train_X.npy')
test_real_X = np.load(f'emb/{args.dataset}/test_X.npy')
train_Y = np.load(f'emb/{args.dataset}/train_Y.npy').astype(int)
test_Y = np.load(f'emb/{args.dataset}/test_Y.npy').astype(int)

train_real_X, train_Y = get_uniform_trainset(
        train_real_X, train_Y, size=args.size)
n_models = train_real_X.shape[2]

### Test HD Accuracy.
encoder = RecordBased(dim=args.dim, n_bundles=train_real_X.shape[1], 
                      n_levels=20, low=-1, high=1)
agg = Aggregator(n_models=n_models, dim=encoder.dim)

train_HD_X = np.stack([
    encoder.encode(train_real_X[:,:,i]
    ) for i in range(n_models)], axis=2)
train_HD_X = agg.aggregate(train_HD_X)
test_HD_X = np.stack([
    encoder.encode(test_real_X[:,:,i]
    ) for i in range(n_models)], axis=2)
test_HD_X = agg.aggregate(test_HD_X)

trainer = HDTrainer(n_models=n_models)
_, train_acc, test_acc = trainer.train(
        train_HD_X, train_Y, test_HD_X, test_Y,
        n_classes=n_classes, optimize='ErrorWeighted')
print(f"HD: {round(test_acc*100, 1)}%")

### Test Others' Accuracy.
train_real_X = np.mean(train_real_X, axis=2)
test_real_X = np.mean(test_real_X, axis=2)
classical_classification(train_real_X, train_Y, test_real_X, test_Y)
