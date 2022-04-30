import torch.nn as nn

class lenet(nn.Module):
    
    def __init__(self, n_classes):
        
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(1, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Tanh())

        self.fc = nn.Sequential(nn.Linear(128, n_classes), nn.Softmax(dim=1))

        self.dim = 128

    def forward(self, x):
        self.z = self.feature(x)
        return self.fc(self.z)

