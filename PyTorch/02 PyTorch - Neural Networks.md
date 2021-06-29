# PyTorch Neural Networks

Author: methylDragon  
Contains a syntax reference for PyTorch  
From various sources, including course notes from SUTD

---



## Credits

A lot of conceptual information was obtained from the [Artificial Intelligence course taught by Prof. Dorien Herremans](https://dorienherremans.com/content/ai-course-2021).

I've added a lot of personal notes and other relevant information from different tutorials and the PyTorch documentation where appropriate. I've also added links and sources to the best of my ability. Hope this reference helps (:



## Pre-Requisites

**Assumed Knowledge**

- Python3
- [NumPy](https://github.com/methylDragon/python-data-tools-reference/blob/master/Numpy/01%20Numpy%20Basics.md) (Highly recommended to have)
- Preceding parts of this reference

**Environment**

- PyTorch (preferably with GPU support)

  - To check if GPU is supported

    ```python
    # Import PyTorch and other libraries
    import torch
    import numpy as np
    from tqdm import tqdm
    
    print("PyTorch version:")
    print(torch.__version__)
    print("GPU Detected:")
    print(torch.cuda.is_available())
    ```

- Colab or Jupyter notebook (good to have)



## Introduction

> **Note**
>
> This reference is not intended to be comprehensive, but should give you enough of a foundation to do a lot with PyTorch!
>
> Also, this is probably not the best place to get explanations of the deep learning math, you'd probably want to refer to a course for that...

Now that we've gotten a hang of working with tensors and the autograd, let's learn about neural networks on PyTorch!



## Neural Networks

### Linear Layers

A single layer of linear transformations is represented as a `nn.Linear` layer.

This defines the function...
$$
f(x) = Wx + b
$$

```python
torch.nn.Linear(in_features, out_features, bias=True)

# Example
lin = nn.Linear(5, 3)
lin(torch.randn(2, 5))
```

You can then stack the Linear layers to form a neural network. (Note that stacking linear layers doesn't do much if you don't put activation functions on them! If you don't the layers' effects will just stack and become equivalent to one linear transformation...)



#### **Dimensions**

[Doc](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

![image-20210630005207722](assets/02%20PyTorch%20-%20Neural%20Networks/image-20210630005207722.png)

This means that all that really matters is that your innermost dimension matches the number of input features!

- The first dimension N can be treated as your batch_size
- The additional dimensions can be treated as something that the Linear mapping is 'broadcast' over
- The final dimension H is your number of input features

Example:

```python
net = nn.Linear(1, 3) # Map from 1 -> 3

net(torch.rand(2, 1))
# tensor([[ 0.0698,  0.4201, -1.3447],
#         [ 0.1366,  0.3710, -1.2822]], grad_fn=<AddmmBackward>)

net(torch.rand(2, 3, 1))
# tensor([[[ 0.0481,  0.4361, -1.3650],
#          [ 0.2489,  0.2883, -1.1768],
#          [ 0.2662,  0.2756, -1.1606]],
#
#         [[ 0.3478,  0.2155, -1.0841],
#          [ 0.4102,  0.1696, -1.0257],
#          [ 0.0038,  0.4688, -1.4066]]], grad_fn=<AddBackward0>)
```



### Activation Functions

For full list, see the [docs](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)

You can use these to introduce non-linearities. Let's take a look at some below!

![What is ReLU and Softmax? - Quora](assets/02%20PyTorch%20-%20Neural%20Networks/main-qimg-07bc0ec05532caf5ebe8b4c82d0f5ca3)

[Image Source](https://laptrinhx.com/classical-neural-net-why-which-activations-functions-3058390165/)

```python
data = torch.randn(2, 3)

relu = torch.nn.ReLU()
print(relu(data))

tanh = torch.nn.Tanh()
print(tanh(data))

sigmoid = torch.nn.Sigmoid()
print(sigmoid(data))
```

You can also call the activations functionally instead of building a class instance like so

```python
torch.relu(data)
torch.tanh(data)
torch.sigmoid(data)
```



### Loss  Functions

We've seen in the previous section that we can use `BCELoss()`. But PyTorch has a couple more!

There's actually a [whole bunch](https://neptune.ai/blog/pytorch-loss-functions) of losses. I'll just mention the more pertinent ones.

```python
torch.nn.L1Loss() # Mean absolute error
torch.nn.MSELoss() # Mean squared error

torch.nn.CrossEntropyLoss() # Cross Entropy (For multiple classes)
torch.nn.BCELoss() # Binary Cross Entropy (For one binary class) (remember to prepend sigmoid)
torch.nn.KLDivLoss() # KL Divergence
```

>  You can also use the functional equivalents with `torch.nn.functional.cross_entropy_loss()`, for example.



### Optimisers

Gradient descent might be easy to compute for one example, but we preferably want to compute it across **all training examples**. This can be an issue if the number of training examples you have is large, you can run out of memory that way.

There's a bunch of alternate gradient descent algorithms that help solve this! [More Optimisers](https://pytorch.org/docs/stable/optim.html)

```python
torch.optim.SGD(model.parameters(), lr=0.001, ...)
torch.optim.Adam(model.parameters(), lr=0.001, ...)
torch.optim.Adagrad(model.parameters(), lr=0.001, ...)
torch.optim.Adadelta(model.parameters(), lr=0.001, ...)
torch.optim.RMSProp(model.parameters(), lr=0.001, ...)
```



### Regularisation

To improve generalisation of model on unseen data:

- Dropout
- Early stopping
- L1 and L2
- Data augmentation
  - Add prior knowledge to the model



#### **Dropout**

During training, **randomly set some weights to zero**. This makes the network not rely too much on any one node.

```python
self.drop_layer = torch.nn.Dropout(p=0.5)

# Or you can do it as a function...
# But you must set training=True, otherwise nothing will happen
torch.nn.functional.dropout(data, 0.5, training=True)
```

![image-20210627041903528](assets/02%20PyTorch%20-%20Neural%20Networks/image-20210627041903528.png)



#### **L1 or L2 Regularisation**

There's two ways to do this! (Or at least, two ways for L2, one way for L1.)

```python
# The easy way for L2, use the weight decay parameter in optimisers
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
```

The other way is to do it manually in our training block. Use `torch.norm()`!:

```python
lambda_1, lambda_2 = 0.5, 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# Training...
optimizer.zero_grad()
outputs, layer1_out, layer2_out = model(inputs)
cross_entropy_loss = F.cross_entropy(outputs, targets)

all_linear1_params = torch.cat([x.view(-1) for x in model.linear1.parameters()])
all_linear2_params = torch.cat([x.view(-1) for x in model.linear2.parameters()])

# Compute regularisation
l1_reg = lambda_1 * torch.norm(all_linear1_params, 1)
l2_reg = lambda_2 * torch.norm(all_linear2_params, 2)

loss = cross_entropy_loss + l1_reg + l2_reg
loss.backward()
optimizer.step()
```



### Building Your Own Neural Network

Again, subclass from `nn.Module`, and then define your network layers in `__init__()`.

Then, you need to define how a forward pass works with a `forward()` method. This is where we include activation functions and determine how data flows from layer to layer!

```python
import torch.nn as nn
import torch.nn.functional as F

# Here, our neural network input is 28*28, and our output is 10
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)
```

Then you can instantiate it and print it if necessary.

```python
net = Net()
print(net)

# Net(
#   (fc1): Linear(in_features=784, out_features=200, bias=True)
#   (fc2): Linear(in_features=200, out_features=200, bias=True)
#   (fc3): Linear(in_features=200, out_features=10, bias=True)
# )
```



#### **Sequential**

Notice in the previous section that we had to manually link the layers in the `forward()` method earlier. We can make use of the `nn.Sequential()` method instead to bypass it!

You can pass in a list of modules, or an `OrderedDict`.

```python
# Source: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
# Comments modified

# Use Sequential to create a small model.
# When model is run, inputs will be cascaded through the modules
# in order: Conv -> ReLU -> Conv -> ReLU
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

# Using Sequential with OrderedDict. This is functionally the
# same as the above code
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
```

Putting it together... (Note that here we are using `nn.ReLU()` instead of `F.relu(x)`)

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(28 * 28, 200)),
                ('ReLU', nn.ReLU()),
                ('fc2' , nn.Linear(200, 200)),
                ('ReLU', nn.ReLU()),
                ('fc3', nn.Linear(200, 10))
            ])
        )

    def forward(self, x):
        x = model(x)
        return F.log_softmax(x)
```



#### **Feedforward Neural Network**

Here's a generic NN template to build off of. This one manually links the layers, but you could always use Sequential like above.

> **Parameters**
>
> - input_size: Dimensionality of input feature vector
> - num_classes: The number of classes in the classification problem
> - num_hidden: The number of hidden (intermediate) layers to use
> - hidden_dim: The size of each of the hidden layers
> - dropout: The proportion of units to drop out after each layer

> **Module List**
>
> [Docs](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)
>
> Registers appended modules to allow exposure to module methods.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

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
```



#### **Model Settings**

You can change some model configurations by calling some methods!

```python
net = FeedForwardNN(...)

# Set to GPU mode
net.cuda()
next(net.parameters()).is_cuda # To check

# Set to inference only (the following are equivalent)
# Affects modules like dropout, batch normalisation, etc. (but not gradient tracking)
net.eval()
net.eval(True)
net.train(False)

# Set to training (the following are equivalent)
# Undoes inferencing mode
net.train()
net.train(True)
net.eval(False)
```

>**On Training and Evaluation mode**
>
>![image-20210630020010846](assets/02%20PyTorch%20-%20Neural%20Networks/image-20210630020010846.png)
>
>[Source](https://stackoverflow.com/a/66843176)



#### **Turning Off Gradient Tracking**

Sometimes you want to turn off gradient tracking (e.g. during inferences, for speedups.) In these cases, use `torch.no_grad()`.

So, for example, for inferencing...

```python
model.eval() # Set to eval mode

with torch.no_grad():
    out_data = model(data)

# Then to go back to training
model.train()
```



#### **Training**

Suppose we have a model that outputs un-normalised scores over 4 classes. We run it with a batch of 3.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, ...)
labels = torch.LongTensor([1, 0, 3])

# CrossEntropyLoss
cross_entropy = nn.CrossEntropyLoss()

# Run one training eposide
optimizer.zero_grad()

model_output = model(x) # (3, 4): 3 examples, 4 classes
avg_loss = cross_entropy(model_output, labels) # Loss, averaged over each batch element
avg_loss.backward() # Backprop

optimizer.step()

# If you want you can look at the gradient of the model_output
model_output.grad
```



#### **Example: Classification with FashionMNIST**

We're using the `FeedforwardNN` class from above!

```python
# Modified from SUTD

import torchvision
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

## LOAD DATA ##
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
    print(next(fashionmnist_ffnn_clf.parameters()).is_cuda) # True if succeeded

nll_criterion = nn.NLLLoss() # Loss
ffnn_optimizer = optim.SGD(fashionmnist_ffnn_clf.parameters(), # Optimizer
                           lr=0.1, momentum=0.9)

## TRAIN ##
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
    
    ## EVALUATE ##
    if num_iter % 500 == 0: # Evaluate every 500 gradient updates
        evaluate_net(fashionmnist_ffnn_clf, test_dataloader) # See below for evaluation function
```

**Evaluation Function**

```python
# Note: This is just for this example, customise it if you want to use it!
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
```



### Recurrent Neural Networks (RNN)

Ok, now that we have some basic foundation, let's start using more complicated neural network units!



#### **Vanilla RNN**

[Docs](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)

Your bog standard RNN. Propagates a hidden state in the network from step to step.

> ![Understanding hidden memories of recurrent neural networks | the morning  paper](assets/02%20PyTorch%20-%20Neural%20Networks/rnnvis-fig-2.jpeg)
>
> [Source](https://blog.acolyer.org/2019/02/25/understanding-hidden-memories-of-recurrent-neural-networks/)

```python
# Where hidden_size is the number of features of the hidden state h
rnn = nn.RNN(input_size, hidden_size, out_size)
```



#### **LSTM**

[Docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

> ![Pytorch中nn.LSTM与nn.LSTMCell_XiaobaiLaplace的博客-CSDN博客_nn.lstmcell](assets/02%20PyTorch%20-%20Neural%20Networks/20201120150748488.png)
>
> [Image Source](https://blog.csdn.net/XiaobaiLaplace/article/details/109842576)
>
> Where h is the hidden state, c is the cell state.

- input_size: Number of features in input

- hidden_size: Number of features in the hidden state

- num_layers: Number of recurrent layers (e.g. num_layers=2 is two LSTM units, with the first feeding its output to the second.)

  - > num_layers example with 2 layers
    >
    > ![LSTM的num_layers是什么意思？ 已解决- MXNet Gluon - MXNet / Gluon 论坛](assets/02%20PyTorch%20-%20Neural%20Networks/21fd219f0964b5981f5534eac93fa30ab9be3460.jpg)
    >
    > [Image Source](![LSTM的num_layers是什么意思？ 已解决- MXNet Gluon - MXNet / Gluon 论坛](assets/02%20PyTorch%20-%20Neural%20Networks/21fd219f0964b5981f5534eac93fa30ab9be3460.jpg))

> **Outputs**
>
> You'll get a sequence of outputs, one at each sequence step fed in!

```python
rnn = nn.LSTM(10, 20, 2) # (input_size, hidden_size, num_layers)

input = torch.randn(5, 3, 10) # Here the input is provided as (sequence_length, batch_size, input_size)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)

# Output shape: (sequence_length, batch_size, input_size * bidirectional)
output, (hidden_state, cell_state) = rnn(input, (h0, c0))

output # Output
output[:, -1] # Final output
```

If your input specifies `batch_size` first, use `batch_first=True`.

>  Notice that `batch_first=True` does not change the `input_size` or `hidden_size` parameters!

```python
# If your input is (batch_size, sequence_length, input_size), use batch_first=True
# Output shape: (batch_size, sequence_length, input_size * bidirectional)
rnn = nn.LSTM(10, 20, 2, batch_first=True)
```

You can do bidirectional LSTMs too!

```python
# You can do a bidirectional LSTMs by passing in
rnn = nn.LSTM(10, 20, 2, bidirectional=True)
```



#### **LSTM Cell**

[Docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html)

`LSTMCell` differs from `LSTM` in how you use it.

- `LSTMCell` is applied manually one by one on timesteps of the input
- `LSTM` is instead applied on the entire input sequence (in an optimised for loop using CuDNN)

This means you can use `LSTMCell` in a different context other than operating over a sequence! But it's more complicated because you're fiddling with timesteps one by one yourself.

```python
# hx is the hidden state, cx is the cell state
rnn = nn.LSTMCell(10, 20) # (input_size, hidden_size)
input = torch.randn(2, 3, 10) # (time_steps, batch, input_size)
hx = torch.randn(3, 20) # (batch, hidden_size)
cx = torch.randn(3, 20) # (batch, hidden_size)

output = []

for i in range(input.size()[0]):
    hx, cx = rnn(input[i], (hx, cx))
    output.append(hx)

output = torch.stack(output, dim=0)
```



#### **Example: Sentiment Analysis with Bidirectional LSTMs [WIP]**

[Source](https://github.com/bentrevett/pytorch-sentiment-analysis)

