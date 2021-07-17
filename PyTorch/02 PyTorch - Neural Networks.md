# PyTorch Neural Networks

Author: methylDragon  
Contains a syntax reference for PyTorch  
From various sources, including course notes from SUTD

---



## Credits

A lot of conceptual information and some of the template sources was obtained from the [Artificial Intelligence course taught by Prof. Dorien Herremans](https://dorienherremans.com/content/ai-course-2021).

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
torch.nn.functional.relu(data)
torch.nn.functional.tanh(data)
torch.nn.functional.sigmoid(data)
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

> #### **L1 Regularisation (Lasso/L1)**
>
> (Aka Lasso or L1 norm.)
>
> Add a regularisation term penalises large parameters. (Drives insignificant features to 0.)
>
> $\theta$ are the weights. $\lambda$ is the penalty term/regularisation parameter.
>
> ![image-20210627194902792](assets/02%20PyTorch%20-%20Neural%20Networks/image-20210627194902792.png)
>
> 
>
> #### **L2 Regularisation (Ridge)**
>
> (Aka Ridge or L2)
>
> Works like L1, but instead of forcing parameters to 0, makes them smaller. (Non-sparse solution.)
>
> ![image-20210627195026635](assets/02%20PyTorch%20-%20Neural%20Networks/image-20210627195026635.png)
>
> - This is **not robust to outliers** as the regularisation parameter will try to penalise the weights more when the loss is too high
> - Performs better when all the input features influence the output and all the weights are roughly equal size

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
          nn.Conv2d(1, 20, 5),
          nn.ReLU(),
          nn.Conv2d(20, 64, 5),
          nn.ReLU()
        )

# Using Sequential with OrderedDict. This is functionally the
# same as the above code
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1, 20, 5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20, 64, 5)),
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

> There's a general training function found in `/Resources/Examples/utils/train.py` if you need it!

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



#### **Template: Feedforward Neural Network**

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

# Call on input as normal
rnn(x)
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

> Bidirectional RNNs are just 2-layer RNNs, connected forward and backward.
>
> ![image-20210630031003198](assets/02%20PyTorch%20-%20Neural%20Networks/image-20210630031003198.png)
>
> [Image Source](http://colah.github.io/posts/2015-09-NN-Types-FP/)
>
> So it can look at the sequence from front-to-back, and back-to-front, and learn from it!

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



#### **Template: Bidirectional LSTM**

Here's a generic Bidirectional LSTM template to build off of. This one manually links the layers, but you could always use Sequential like above.

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
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim=100, hidden_dim=256, output_dim=1, n_layers=2,
                 bidirectional=True, dropout=0.5, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(input_dim,
                                      embedding_dim,
                                      padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text)) # Map text to embedding

        # Pack sequence
        # Note: We move text_lengths to cpu due to a small bug
        # https://github.com/pytorch/pytorch/issues/43227
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu()
        )

        packed_output, (hidden, cell) = self.rnn(packed_embedded) # Feedforward

        # Unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        hidden = self.dropout(torch.cat((hidden[-2,:,:],
                                         hidden[-1,:,:]),
                                        dim = 1))

        return self.fc(hidden)
```



### Convolutional Neural Networks (CNN)

We'll just talk about 2D convolutions here, but [1D convolution](https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d) is also possible (e.g. for word embeddings!)



#### **Conv2D**

Does convolutions! It's good for image related machine learning tasks, but you can also use it for anything you can represent using 2D matrices, like music (via spectrogram images)!

![File:2D Convolution Animation.gif](assets/02%20PyTorch%20-%20Neural%20Networks/2D_Convolution_Animation.gif)

[Image Source](https://commons.wikimedia.org/wiki/File:2D_Convolution_Animation.gif) | [More animations](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)

```python
# Doc: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
# Single convolutional layer
cnn = nn.Conv2d(input_channels, output_channels, kernel_size)

# Call on 2D input as normal
cnn(x)

# There are more convolution layers, like transposed convolution, which upscales the output!
# See animations for more information
trans_cnn = nn.ConvTranspose2d(input_channels, output_channels, kernel_size)
```

> For examples of the `stride`, `padding`, and `dilation` parameters, see [this link for more animations](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md#transposed-convolution-animations).
>
> But in short:
>
> - **Stride** is how much the kernel moves per step
> - **Padding** pads the input image with 0s on the borders (it'll pad with 0s)
> - **Dilation** 'spreads' out the convolution kernel 
>
> Furthermore:
>
> > The parameters `kernel_size`, `stride`, `padding`, `output_padding` can either be:
> >
> > > - a single `int` – in which case the same value is used for the height and width dimensions
> > > - a `tuple` of two ints – in which case, the first int is used for the height dimension, and the second int for the width dimension
> >
> > [Docs](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)



#### **Dimensions**

If you have more than 1 input channel, your kernel will be multi-dimensional (this is how a 1x1 kernel still makes sense!)

<img src="assets/02%20PyTorch%20-%20Neural%20Networks/image-20210717181347754.png" alt="image-20210717181347754" style="zoom:50%;" />

[Image Source](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)

Then, you'll have one kernel learnt **per** output channel!

The kernel weights per kernel then will be: `kernel_height * kernel_width * in_channels`, for n kernels!!



#### **Activations**

Convolutional layers can flow into activation functions as per normal (as if they were linear layers).

So you can use all your [good friends](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)



#### **Pooling, and Flattening**

> **Pooling**
>
> ![image-20210717182225513](assets/02%20PyTorch%20-%20Neural%20Networks/image-20210717182225513.png)
>
> [Image Source](https://ai.plainenglish.io/pooling-layer-beginner-to-intermediate-fa0dbdce80eb)

```python
# Pooling (there's a lot more)
nn.MaxPool2d(kernel_size)
nn.AvgPool2d(kernel_size)
nn.MaxUnpool2d(kernel_size) # All non-max values are set to 0

# Flattening
out = out.view(out.size(0), -1) 
```



#### **Pre-Processing**

Use `torchvision.transforms` to perform image transformations for pre-processing and `PIL` to open images!

[See all transforms here](https://pytorch.org/vision/stable/transforms.html), or see a [short tutorial with visualisations](https://www.analyticsvidhya.com/blog/2021/04/10-pytorch-transformations-you-need-to-know/)

```python
import torchvision.transforms as transforms
from PIL import Image
import json, io, requests, string

# Create a chain of preprocessing transforms using transforms.Compose
preprocessFn = transforms.Compose(
    [transforms.Resize(256),  # 1. Resize smallest side to 256.
     transforms.CenterCrop(224), # 2. Crop the center 224x224 pixels.
     transforms.ToTensor(), # 3. Convert to pytorch tensor.
     transforms.Normalize(mean = [0.485, 0.456, 0.406],  # normalize.
                          std = [0.229, 0.224, 0.225])])

response = requests.get(img_url)
img_pil = Image.open(io.BytesIO(response.content))

# Unsqueeze adds a dummy batch dimension needed to pass through the model.
input_img =  preprocessFn(img_pil).unsqueeze(0)
predictions = cnn_model(input_img)
```



#### **Using Pre-trained Models**

There are many pre-trained computer vision models out there for use already! You can very easily use them to obtain image features to do transfer learning on, saving massive amounts of time and compute!

```python
import torchvision.models as models

# https://arxiv.org/abs/1512.00567 [Re-thinking the Inception Architecture]
inception_model = models.inception_v3(pretrained=True)

# https://arxiv.org/abs/1512.03385 [Residual Networks]
resnet_model = models.resnet50(pretrained=True)


# Then you can simply call them as necessary
resnet_model(image)
```

You just saw how to use pre-trained models in their entirety, but you can also use the outputs of the hidden layers (so you can do transfer learning!)
