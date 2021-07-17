# Modified from SUTD and https://github.com/bentrevett/pytorch-sentiment-analysis

# LeNet Implementation, run on FashionMNIST
# https://ieeexplore.ieee.org/abstract/document/726791
# PDF: http://www.cs.virginia.edu/~vicente/deeplearning/readings/lecun1998.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
import torch.optim as optim

import utils.train # Local utilities! (:


# MODEL ========================================================================
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv_feature_net = nn.Sequential( # Convolution
          nn.Conv2d(1, 32, 5),
          nn.ReLU(),
          nn.MaxPool2d(2), # Kernel size 2, no stride
          nn.Conv2d(32, 64, 5),
          nn.ReLU(),
          nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential( # Linear layers
          nn.Linear(64*4*4, 120),
          nn.ReLU(),
          nn.Linear(120, 84),
          nn.ReLU(),
          nn.Linear(84, 10) # No softmax here! nn.CrossEntropy does it for us
        )

    def forward(self, x):
        features = self.conv_feature_net(x)

        # This flattens the output of the previous layer into a vector.
        features = features.view(features.size(0), -1)

        return self.classifier(features)


# TRAINING UTILITIES ===========================================================
if __name__ == "__main__":
    # Removes the need to call F.to_image ourselves.
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the training, and validation datasets.
    trainset = FashionMNIST(root = './data', train = True, transform = transform, download = True)
    valset = FashionMNIST(root = './data', train = False, transform = transform, download = True)

    batchSize = 100
    loss_fn = nn.CrossEntropyLoss()
    learningRate = 5e-2

    cnn_model = LeNet()
    optimizer = optim.Adam(cnn_model.parameters(), lr = 3e-3)
    utils.train.train_model(cnn_model, loss_fn,
                            batchSize, trainset, valset,
                            optimizer,
                            num_epochs=5)
