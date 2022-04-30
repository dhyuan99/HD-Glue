import os
from utils import get_network
from collections import namedtuple
import numpy as np
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        '-dataset', type=str, required=True, 
        help='the dataset you want to work on')
args = parser.parse_args()

train_X = torch.Tensor(np.load(f'data/{args.dataset}/train_X.npy'))
train_Y = torch.Tensor(np.load(f'data/{args.dataset}/train_Y.npy'))
test_X = torch.Tensor(np.load(f'data/{args.dataset}/test_X.npy'))
test_Y = torch.Tensor(np.load(f'data/{args.dataset}/test_Y.npy'))

train_emb, test_emb = [], []
for net_ in os.listdir(f'networks/{args.dataset}'):
    if net_.startswith('.'):
        # ignore the .gitignore file.
        continue
    
    net_args = namedtuple('args', ['dataset', 'net'])
    net_args.net = net_.strip('.pth')
    net_args.gpu = False
    net_args.dataset = args.dataset
    net = get_network(net_args)
    net.load_state_dict(torch.load(
        f'networks/{args.dataset}/{net_}', map_location=torch.device('cpu')))
    net.eval()
    with torch.no_grad():
        net(train_X)
    train_emb.append(net.z.detach().numpy())
    with torch.no_grad():
        net(test_X)
    test_emb.append(net.z.detach().numpy())

train_emb = np.stack(train_emb, axis=2)
test_emb = np.stack(test_emb, axis=2)

np.save(f'emb/{args.dataset}/train_X.npy', train_emb)
np.save(f'emb/{args.dataset}/train_Y.npy', train_Y)
np.save(f'emb/{args.dataset}/test_X.npy', test_emb)
np.save(f'emb/{args.dataset}/test_Y.npy', test_Y)
