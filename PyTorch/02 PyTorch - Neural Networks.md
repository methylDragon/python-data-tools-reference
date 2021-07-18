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



#### **Not Training A Portion of the Network**

[Source](https://discuss.pytorch.org/t/how-to-train-a-part-of-a-network/8923/6)

Sometimes you don't want to train parts of a network. For all of these portions...

1. Call them with `torch.no_grad()` on the forward pass to avoid accumulating gradients (and wasting space)

   ```python
   with torch.no_grad():
       out = subnet(inputs)
       
   # Or you can set the subnet to eval mode
   subnet.eval()
   
   # Or you can set every parameter in the subnet to not require gradients
   for param in subnet.parameters():
       param.requires_grad = False
   ```

2. Ensure that you only pass the parameters to be trained to the optimiser

   ```python
   optim.Adam(model.sub_network_to_train.parameters(), ...)
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

You can also run pre-processing transforms if you load a dataset!

If you have a folder of images somewhere, you can load the data using an `ImageFolder` instance, running pre-processing using `torchvision.transforms`!

```python
train_dataset = datasets.ImageFolder("train-uva", train_transform)
val_dataset = datasets.ImageFolder("val-uva", test_transform)
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

You can either alter the network directly (and retrain it)...

```python
# Change the last layer, and then remember to retrain it later on!
resnet_model.fc = nn.Linear(2048, len(train_dataset.classes))
```

Or truncate the network and use the weights (without retraining it.) This uses the network as a feature extractor rather than a classifier, you can then connect it to other parts of your network!

```python
import torchvision.models as models
import torch.nn as nn

resnet_model = models.resnet50(pretrained = True)

class FeatureExtractor(nn.Module):
  def __init__(self, resnet_model):
    super(FeatureExtractor, self).__init__()

    # Truncate the network, removing the last layer
    self.truncated_resnet = nn.Sequential(*list(resnet_model.children())[:-1]).cuda()
  def forward(self, x):
    feats = self.truncated_resnet(x)
    
    # Then discard the height and width dimensions using tensor.view.
    return feats.view(feats.size(0), -1)

# Let's test the feature extractor to see what size are the outputs
# right before the layer we eliminated.
feature_extractor = FeatureExtractor(resnet_model)
dummy_inputs = torch.zeros(10, 3, 224, 224).cuda()
dummy_outputs = feature_extractor(dummy_inputs)
print(dummy_outputs.shape)
```



**Example: Show Image and Top 5 Predictions**

Source: SUTD

```python
my_image, my_label = val_dataset[231]

scene_model.eval()
scene_model.load_state_dict(torch.load('scene_model_weights.pth'))

predictions = scene_model(my_image.cuda().unsqueeze(0))

preds = predictions.data.cpu().softmax(dim = 1)
probs, indices = (-preds).sort()
probs = (-probs).numpy()[0][:5]; indices = indices.numpy()[0][:5]
preds = ['P[\"' + val_dataset.classes[idx] + '\"] = ' + ('%.6f' % prob) \
         for (prob, idx) in zip(probs, indices)]

plt.figure()
plt.title("True-Category: " + val_dataset.classes[my_label] + "\n\n" +
          "\n".join(preds))

# Undo normalization of the pixel values.
for t, m, s in zip(my_image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
  t.mul_(s).add_(m)

# Re-arrange dimensions so it is height x width x channels.
plt.imshow(my_image.transpose(0,2).transpose(0,1));
plt.grid(False); plt.axis('off');
```



## Variational Autoencoders (VAE)

> A variational autoencoder can be defined as being an autoencoder whose training is regularised to avoid overfitting and ensure that the latent space has good properties that enable generative process.
>
> This is done by encoding the input as a **distribution** in the latent space.
>
> [Source](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)

![image-20210718050008117](assets/02%20PyTorch%20-%20Neural%20Networks/image-20210718050008117.png)

[Image Source](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)



#### **Encoder and Decoder**

> **Note**: All the code in this section is basically just taken from the SUTD AI course...

> **Note**: This is for MNIST, with 28x28 grayscale images!!
>
> If you wan to implement a VAE for yourself, you'll need to change the dimensions!



**Encoder**

> Kernel size 4 is to mitigate biasing problems [as described here](https://distill.pub/2016/deconv-checkerboard/)

```python
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.fc_mu = nn.Linear(in_features=c*2*7*7, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c*2*7*7, out_features=latent_dims)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        
        # Here we use use x for mu and for variance! 
        x_mu, x_logvar = self.fc_mu(x), self.fc_logvar(x)
        # Note: We don't calculate x_logvar from x_mu, but use x instead!! 
        return x_mu, x_logvar
```

**Decoder**

```python
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*2*7*7)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), capacity*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x   
```



#### **VAE Model**

```python
class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        # Our encoder output consists of x_mu and x_logvar
        latent_mu, latent_logvar = self.encoder(x)
        
        # we sample from the distributions defined by mu and logvar
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_() # Get standard deviation
            eps = torch.empty_like(std).normal_() # Define normal distribution
            return eps.mul(std).add_(mu) # Sample from normal distribution
        else:
            return mu
```



#### **VAE Loss**

```python
variational_beta = 1

def vae_loss(recon_x, x, mu, logvar):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    reconstruction_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    
    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return reconstruction_loss + variational_beta * kldivergence
    
    
vae = VariationalAutoencoder()
```



#### **Train**ing

Then we basically train it, using the input as the label, since we want the autoencoder to learn to reconstruct the input after forcing it through a bottleneck.

```python
optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-5)

vae.train()
train_loss_avg = []

print('Training ...')
for epoch in range(num_epochs):
    train_loss_avg.append(0)
    num_batches = 0
    
    for image_batch, _ in train_dataloader:
        image_batch = image_batch.to(device)

        # VAE reconstruction
        image_batch_recon, latent_mu, latent_logvar = vae(image_batch)        
        loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)
        
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()
        
        train_loss_avg[-1] += loss.item()
        num_batches += 1
        
    train_loss_avg[-1] /= num_batches
    print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))
```



#### **Visualise Reconstructions**

![image-20210718051609745](assets/02%20PyTorch%20-%20Neural%20Networks/image-20210718051609745.png)

```python
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torchvision.utils

vae.eval()

def to_img(x):
    x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualise_output(images, model):
    with torch.no_grad():
        images = images.to(device)
        # fetch the generated images by calling the model: 
        images, _, _ = model(images)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()

images, labels = iter(test_dataloader).next()

print('Original images')
show_image(torchvision.utils.make_grid(images[1:50],10,5))
plt.show()

print('VAE reconstruction:')
visualise_output(images, vae)
```



#### **Interpolate in Latent Space**

![image-20210718051924285](assets/02%20PyTorch%20-%20Neural%20Networks/image-20210718051924285.png)

```python
vae.eval()

def interpolation(lambda1, model, img1, img2):
    with torch.no_grad():
        # latent vector of first image
        img1 = img1.to(device)
        latent_1, _ = model.encoder(img1)

        # latent vector of second image
        img2 = img2.to(device)
        latent_2, _ = model.encoder(img2)

        # we interpolate the z_1 to z_2 with out lambda1
        inter_latent = lambda1* latent_1 + (1- lambda1) * latent_2

        # reconstruct interpolated image, just use the decoder
        inter_image = model.decoder(inter_latent)
        inter_image = inter_image.cpu()

        return inter_image

# sort part of test set by digit
digits = [[] for _ in range(10)]
for img_batch, label_batch in test_dataloader:
    for i in range(img_batch.size(0)):
        digits[label_batch[i]].append(img_batch[i:i+1])
    if sum(len(d) for d in digits) >= 1000:
        break;

# interpolation lambdas, create an equally spaced range
lambda_range=np.linspace(0,1,10)

# setup the plot
fig, axs = plt.subplots(2,5, figsize=(15, 6))
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()

for ind,l in enumerate(lambda_range):
    inter_image=interpolation(float(l), vae, digits[7][0], digits[1][0])
   
    inter_image = to_img(inter_image)    
    image = inter_image.numpy()
    
    # plot the interpolated image
    axs[ind].imshow(image[0,0,:,:], cmap='gray')
    axs[ind].set_title('lambda_val='+str(round(l,1)))
plt.show() 
```



#### **Visualise 2D Latent Space**

![image-20210718052112892](assets/02%20PyTorch%20-%20Neural%20Networks/image-20210718052112892.png)

Trippy

```python
# load a network that was trained with a 2d latent space
if latent_dims != 2:
    print('Please change the parameters to two latent dimensions.')
    
with torch.no_grad():
    # create a sample grid in 2d latent space
    latent_x = np.linspace(-1.5,1.5,20)
    latent_y = np.linspace(-1.5,1.5,20)
    latents = torch.FloatTensor(len(latent_y), len(latent_x), 2)
    for i, lx in enumerate(latent_x):
        for j, ly in enumerate(latent_y):
            latents[j, i, 0] = lx
            latents[j, i, 1] = ly
    latents = latents.view(-1, 2) # flatten grid into a batch

    # reconstruct images from the latent vectors
    latents = latents.to(device)
    image_recon = vae.decoder(latents)
    image_recon = image_recon.cpu()

    fig, ax = plt.subplots(figsize=(10, 10))
    show_image(torchvision.utils.make_grid(image_recon.data[:400],20,5))
    plt.show()
```



### Conditional Generative Adversarial Networks (cGAN)

[Paper](https://arxiv.org/abs/1411.1784)

We can have conditional GANs to generate desired outputs!

- [PyTorch implementation](https://www.kaggle.com/arturlacerda/pytorch-conditional-gan)
- The generator learns to generate a fake sample with a **specific condition or characteristics** rather than a generic sample from unknown noise distribution

![image-20210718052243234](assets/02%20PyTorch%20-%20Neural%20Networks/image-20210718052243234.png)

For a GAN, we'll want to train a generator and a discriminator.

We'll get a cGAN for FashionMNIST!

```python
from torchvision.datasets import FashionMNIST

transform = transforms.Compose([
        transforms.ToTensor(),transforms.Normalize(mean=(0.5), std=(0.5))
])

dataset = FashionMNIST(root='./data', train=True, transform=transform, download=True)
```



#### **Generator and Discriminator**

> **Note**: All the code in this section is basically just taken from the SUTD AI course...
>
> Also, again, the network layer dimensions are for a specific problem (in this case FashionMNIST), do change them as you need.

**Generator**

Note that we could also use a CNN for the discriminator.

```python
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10) # For the label
        
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        z = z.view(z.size(0), 100) # Noise
        c = self.label_emb(labels) # Desired label (the condition)

        # Concatenates z and c, so the condition is given as input to the model appended to z
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)
```



**Discriminator**

```python
 class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(10, 10) # For the label
        
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = x.view(x.size(0), 784) # Image (either fake or real)
        c = self.label_emb(labels) # Label (the condition)

        # Concatenates x and c, so the condition is given as input to the model appended to the image x
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()
```



#### **Training**

The training for a GAN takes place in two distinct, alternating phases.

- GAN training proceeds in alternating periods:
  - The discriminator trains for one or more epochs
  - Then the generator trains for one or more epochs
  - Repeat

> The discriminator performance should get worse and converge to 50%.
>
> But, this presents a problem! The **discriminator feedback gets less useful over time**, and continued training will cause the generator to start training on junk feedback, causing its own quality to collapse.
>
> So GAN convergence is **often a fleeting**, rather than stable state.

```python
def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).cuda() # Random noise
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).cuda())
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    d_optimizer.zero_grad()

    # Train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).cuda())
    
    # Train with fake images
    z = Variable(torch.randn(batch_size, 100)).cuda() # Random noise
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    fake_images = generator(z, fake_labels) # Generate fake images
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).cuda())
    
    # Optimize the sum of both losses
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data
```

Aaand train!

```python
num_epochs = 30
n_critic = 5
display_step = 300

for epoch in range(num_epochs):
    print('Starting epoch {}...'.format(epoch))
    for i, (images, labels) in enumerate(data_loader):
        real_images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Train the generator
        generator.train()
        batch_size = real_images.size(0)
        
        # Train the discriminator
        d_loss = discriminator_train_step(len(real_images), discriminator,
                                          generator, d_optimizer, criterion,
                                          real_images, labels)
        

        g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)

    generator.eval()
    print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))

    z = Variable(torch.randn(9, 100)).cuda() # Random noise
    labels = Variable(torch.LongTensor(np.arange(9))).cuda()

    # Generate a new image for each of the labels based on the label and the noise z
    sample_images = generator(z, labels).unsqueeze(1).data.cpu()

    # Display the images
    grid = make_grid(sample_images, nrow=3, normalize=True).permute(1,2,0).numpy()
    plt.imshow(grid)
    plt.show()
```

