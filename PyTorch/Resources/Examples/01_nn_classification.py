# Modified from SUTD
# Classification with FashionMNIST

import torchvision
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F


# MODEL ========================================================================
class FeedForwardNN(nn.Module):
  def __init__(self, input_size, num_classes, num_hidden, hidden_dim, dropout):
    super(FeedForwardNN, self).__init__()
    assert num_hidden > 0

    # Dropout and Activation
    self.dropout = nn.Dropout(dropout)
    self.nonlinearity = nn.ReLU()

    # Hidden layers
    self.hidden_layers = nn.ModuleList([])
    self.hidden_layers.append(nn.Linear(input_size, hidden_dim))

    for i in range(num_hidden - 1):
      self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

    # Output
    self.output_projection = nn.Linear(hidden_dim, num_classes)

  # input is of shape (batch_size, input_size)
  def forward(self, x):
    for hidden_layer in self.hidden_layers:
      x = hidden_layer(x) # Apply hidden layer
      x = self.dropout(x) # Apply dropout
      x = self.nonlinearity(x) # Apply

    out = self.output_projection(x) # Map output

    # Softmax output to map to log-probability distribution over classes for each example
    out_distribution = F.log_softmax(out, dim=-1)
    return out_distribution


# EVALUATION ===================================================================
def evaluate_net(net, test_data):
    net.eval() # Eval mode
    num_correct, total_examples, total_test_loss = 0, 0, 0

    for (test_images, test_labels) in test_data:
        reshaped_test_images = test_images.view(-1, 784) # Reshape to fit model

        if using_GPU:
            reshaped_test_images = reshaped_test_images.cuda()
            test_labels = test_labels.cuda()

        predicted = fashionmnist_ffnn_clf(reshaped_test_images) # Forward pass

        # Loss is averaged, multiply by batch size
        total_test_loss += nll_criterion(predicted, test_labels) * test_labels.size(0)

        _, predicted_labels = torch.max(predicted.data, 1) # Get predicted label

        total_examples += test_labels.size(0)

        num_correct += torch.sum(predicted_labels == test_labels)
        accuracy = 100 * num_correct / total_examples
        average_test_loss = total_test_loss / total_examples

        print("Iteration {}. Test Loss {}. Test Accuracy {}.".format(
            num_iter, average_test_loss, accuracy))

    net.train() # Back to train mode


if __name__ == "__main__":
    # LOAD DATA ================================================================
    train_dataset = FashionMNIST(root='./torchvision-data',
                                 train=True,
                                 transform=torchvision.transforms.ToTensor(),
                                 download=True)

    test_dataset = FashionMNIST(root='./torchvision-data', train=False,
                                transform=torchvision.transforms.ToTensor())

    batch_size = 64

    # Dataloader automatically reshapes out data for us
    # We went from dataset elements of shape (1, 28, 28) and labels of shape (1)
    # To elements of (64, 1, 28, 28) and labels of shape (64) (batch size accounted for)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size)

    ## MODEL ##
    fashionmnist_ffnn_clf = FeedForwardNN(input_size=784, num_classes=10,
                                          num_hidden=2,
                                          hidden_dim=512, dropout=0.2)

    if using_GPU: # Shift to GPU if necessary
        fashionmnist_ffnn_clf = fashionmnist_ffnn_clf.cuda()
        print(next(fashionmnist_ffnn_clf.parameters()).is_cuda) # True if ok

    nll_criterion = nn.NLLLoss() # Loss
    ffnn_optimizer = optim.SGD(fashionmnist_ffnn_clf.parameters(), # Optimizer
                               lr=0.1, momentum=0.9)


    # TRAIN ====================================================================
    num_epochs = 10
    num_iter = 0 # Counter for iters done

    for epoch in range(num_epochs):
      print("Starting epoch {}".format(epoch + 1))

      for (images, labels) in train_dataloader:
        reshaped_images = images.view(-1, 784) # Reshape to fit model

        if using_GPU:
          reshaped_images = reshaped_images.cuda()
          labels = labels.cuda()

        predicted = fashionmnist_ffnn_clf(reshaped_images)
        batch_loss = nll_criterion(predicted, labels)

        ffnn_optimizer.zero_grad()
        batch_loss.backward()
        ffnn_optimizer.step()

        num_iter += 1 # Increment gradient update counter

        # EVALUATE =============================================================
        if num_iter % 500 == 0: # Evaluate every 500 gradient updates
            evaluate_net(fashionmnist_ffnn_clf, test_dataloader)
