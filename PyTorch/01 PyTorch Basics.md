# PyTorch Reference

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

> PyTorch enables fast, flexible experimentation and efficient production through a user-friendly front-end, distributed training, and ecosystem of tools and libraries.

It can be used as a replacement for numpy with GPU support and deep learning routines!

This reference is not intended to be comprehensive, but should give you enough of a foundation to do a lot with PyTorch!

Also, this is probably not the best place to get explanations of the deep learning math, you'd probably want to refer to a course for that...



## Bonus: Google Colab

Lots of tutorials run stuff on Google Colab or Jupyter notebooks. Here's just a bunch of handy code snippets to do basic stuff with it.



### Enable GPU/TPU Acceleration

`Edit > Notebook settings`



### Load Files into Colab

To see how to load files into PyTorch tensors, check the Tensor section!

```python
# From raw link (< 25mb) (Including Github!)
url = 'some_raw_direct_link'
df_1 = pd.read_csv(url)

# From local drive
from google.colab import files
uploaded = files.upload() # It'll open a file dialog

# From Google Drive
from google.colab import drive
drive.mount('/content/drive') # Use this to auth
url = 'google_drive_path' # Get this from copying the path of your Google Drive data
df_1 = pd.read_csv(url)
```



## Tensors

[Tutorial](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py) | [API Docs](https://pytorch.org/docs/master/tensors.html)



### Tensor Overview

Tensors are the basic units of computation in PyTorch. You can think of them like numpy arrays.



### Tensor Types

You can get a full [list of Tensor types here](https://pytorch.org/docs/master/tensors.html), but generally the tensors that are the most commonly used are:

| Data Type          | CPU Tensor Type     | GPU Tensor Type          | NumPy Type |
| ------------------ | ------------------- | ------------------------ | ---------- |
| 32-bit Float       | `torch.FloatTensor` | `torch.cuda.FloatTensor` | `float32`  |
| 8-bit Unsigned Int | `torch.ByteTensor`  | `torch.cuda.ByteTensor`  | `uint8`    |
| 64-bit Int         | `torch.LongTensor`  | `torch.cuda.LongTensor`  | `int64`    |

You generally want to use `FloatTensor` by default, unless your data is an integer (`LongTensor`) or bits (`ByteTensor`)

> `torch.Tensor` is an alias of `torch.FloatTensor`!



### Concepts

#### **Dimensions**

<img src="assets/01%20PyTorch%20Basics/tensor-examples.jpg" alt="PyTorch Tensor Basics - KDnuggets" width="50%" />

[Image Source](https://www.kdnuggets.com/2018/05/pytorch-tensor-basics.html)

The first index can always be interpreted as 'rows'. They function just like numpy ndarrays!

More intuitively, each additional dimension is one more layer of access.

Example:

```python
torch.rand(1)
# tensor([0.0713])

torch.rand(1, 2)
# tensor([[0.1710, 0.6371]])

torch.rand(1, 2, 3)
# tensor([[[0.1652, 0.4824, 0.6642],
#          [0.1946, 0.2631, 0.3563]]])
```

Notice how the innermost dimension is the last one.





#### **Broadcasting**

![array](assets/01%20PyTorch%20Basics/array.jpg)

[Image Source](https://www.tutorialspoint.com/numpy/numpy_broadcasting.htm)

If you have dimension mismatches, broadcasting can occur, where tensors are "stretched" or "padded" to match the dimensions.

You can see in the image above that the b tensor was applied repeatedly on addition.



### Creating Tensors

```python
# Create an UNINITIALISED (Float)Tensor! Values are NOT 0
uninit_float = torch.Tensor(4, 6)

# tensor([[1.8783e+34, 3.0686e-41, 7.0065e-44, 6.8664e-44, 6.3058e-44, 6.7262e-44],
#         [7.5670e-44, 6.3058e-44, 6.7262e-44, 6.8664e-44, 1.1771e-43, 6.7262e-44],
#         [7.1466e-44, 8.1275e-44, 7.4269e-44, 6.8664e-44, 8.1275e-44, 6.7262e-44],
#         [7.5670e-44, 6.4460e-44, 7.9874e-44, 6.7262e-44, 7.2868e-44, 7.4269e-44]])

# From List
some_float_tensor = torch.Tensor([3.2, 4.3, 5.5])

# Pre-filled
rand_float = torch.rand(4, 6) # Random values between 0.0 and 1.0
randn_float = torch.rand(4, 6) # Random values drawn from standard normal distribution N~(0, 1)
ones_float = torch.ones(4, 6)
zeros_float = torch.zeros(4, 6)

# From other tensor's dimensions
x_ones = torch.ones_like(x_data) # Retains the properties of x_data, but all ones
x_rand = torch.rand_like(x_data, dtype=torch.float) # Overrides the datatype of x_data
```



### Loading Files

```python
import torch
import pandas as pd

train = pd.read_csv('train.csv')
train_tensor = torch.tensor(train.to_numpy())
```

> **Note**: This is **not** `Tensor`, it's `tensor`, a different constructor.



### Getting Tensor Info

Note that the `Size` class is actually just a tuple

```python
rand_float = torch.rand(4, 6)

# Get Dimension
rand_float.size() # torch.Size([4, 6])
rand_float.shape # Equivalent

# Get specific dimension
rand_float.size(0)
rand_float.size()[0] # Equivalent, both return 4

# Get Tensor Info
rand_float.type() # torch.FloatTensor
rand_float.dtype() # torch.float32
rand_float.device # cpu

# Get Memory Location (If equal, memory is shared)
rand_float.data_ptr()
```



### Moving to GPU

```python
# Move to GPU
if torch.cuda.is_available():
    tensor_gpu = tensor.to('cuda')
    tensor_gpu = tensor.cuda() # Also works
    
# Move to CPU
tensor_gpu.to('cpu')
tensor_gpu.cpu()
```

You'll get big speedups **as long as you're not explicitly iterating over your tensors**! Leave it to the library to do operations, yeah?



### Shape Ops

#### **Indexing**

Works just like numpy! Maybe, check out the [numpy tutorial](https://github.com/methylDragon/python-data-tools-reference/blob/master/Numpy/01%20Numpy%20Basics.md)?

> **Important Note**: For slices and reshapes, PyTorch (just like NumPy) creates a **view** of the data! That means that data isn't actually copied, so operating on either the source or the reshaped tensor **affects both since they share the same underlying memory**!!!
>
> **Caveat**: Actually, `reshape()` tries to create a view. If it can't it'll end up in a copy... If you need to enforce data-sharing, use `view()`

```python
row = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 8, 9])
# tensor([0., 1., 2., 3., 4., 5., 6., 8., 9.])

square = row.reshape(3, 3)
# tensor([[0., 1., 2.],
#         [3., 4., 5.],
#         [6., 8., 9.]])

# Basic Indexing
row[5] # tensor(5.)
square[0] # tensor([0., 1., 2.])
square[0][0] # tensor(0.)
square[0, 0] # tensor(0.)
```

**Advanced Indexing**

```python
# Multiple Indexing
row[[0, 2, 5]] # Get indexes 0, 2, and 5: 0., 2., 5.

square[(0, 1), (2, 2)] # Get (0, 2) and (2, 2): tensor([2., 5.])
square[0, (1, 2)] # Get (0, 1) and (0, 2), by broadcasting: tensor([1, 2.])

# Indexing with LongTensors
square[torch.LongTensor([[0, 1]])] # This does NOT work like above!
# tensor([[[0., 1., 2.],
#          [3., 4., 5.]]])
```

> **Note**: Multidimensional multi-indexing works on a dimension by dimension basis!!
>
> So, for example, indexing into square with `square[(a, b), (c, d)]` is equivalent to concatenating the results of `square[a, c]` and `square[c, d]`.
>
> **Additional note**: Also beware of broadcasting if you mismatch the dimensions of your multiple indices!



#### **Slicing**

```python
# Slicing
row[:5] # [0., 1., 2., 3., 4.]
square[:1] # [[0., 1., 2.]]
square[:, 1] # Take column index 1: [1., 4., 8.]
```



#### **Reshaping**

```python
# Reshaping
square = row.reshape(3, 3) # Set new shape
# tensor([[0., 1., 2.],
#         [3., 4., 5.],
#         [6., 8., 9.]])

square.reshape_as(row) # Copy shape of existing tensor
# tensor([0., 1., 2., 3., 4., 5., 6., 8., 9.])
```

> **Tip**: You can use `-1` on ONE dimension to have Torch infer the right number for you! But be very careful, it's prone to introducing bugs if dimensions vary, since behaviour will also vary!

Reshapes try to create a **view** of the underlying tensor, which means that data will be shared. If it can't it'll create a copy. If you want to enforce data sharing, use `view()`.



#### **Concatenating**

> **Note**: `axis` is an alias of `dim`

```python
# Flat
tri = torch.Tensor([0, 1, 2])
torch.cat((tri, tri)) # tensor([0., 1., 2., 0., 1., 2.])

# Higher Dimensions
trid = torch.Tensor([[0, 1, 2]]) # Note this is 2D now!
torch.cat((trid, trid), axis=1) # tensor([0., 1., 2., 0., 1., 2.])

torch.cat((trid, trid)) # Default is along axis=0
# tensor([[0., 1., 2.],
#         [0., 1., 2.]])

# Concatenate along a NEW dimension
torch.stack((tri, tri))
```



#### **Adding and Removing Dimensions**

```python
tri = torch.Tensor([0, 1, 2])

# Add dimensions
tri.unsqueeze(0) # Add a dimension in axis 0
# tensor([[0., 1., 2.]])

tri.unsqueeze(1) # Add a dimension in axis 1
# tensor([[0.],
#         [1.],
#         [2.]])

tri.squeeze() # Remove all extra dimensions
tri.squeeze(1) # Remove all extra dimensions along axis 1
```



#### **Transposing**

```python
# Transpose (All the following are equivalent)
torch.transpose(some_tensor, 0, 1) # Swaps dimensions 0 and 1 of some_tensor
some_tensor.transpose(0, 1)
torch.t(some_tensor) # Equivalent to transpose(input, 0, 1) only

# Permute (Tranpose is actually a special case of permute) (All the following are equivalent)
# Suppose some_tensor was (2, 3, 4)
torch.permute(some_tensor, 1, 0, 2) # Makes some_tensor (3, 2, 4)
some_tensor.permute(1, 0, 2)
```



#### **Flattening**

```python
# Flatten and Ravel
# These both return a flattened array!
.ravel() # Returns a view
.flatten() # Returns a copy, so is slower and takes more memory
```



### Tensor Arithmetic

> **Note**: Most operations have an in-place modification variant. Just add an underscore!
>
> Applying operations in-place **modifies the original tensor**, whereas not doing so returns a **copy** of the tensor with the operation executed.
>
> E.g. `add()` vs `add_()`.

```python
tri = torch.Tensor([0, 1, 2])

# Element-wise Addition (All the following are equivalent)
tri + tri # tensor([0., 2., 4.])
torch.add(tri, tri)
tri.add(tri)

# Element-wise Multiplication (All the following are equivalent)
tri * tri # tensor([0., 1., 4.])
torch.multiply(tri, tri)
tri.multiply(tri)

# Sum all elements in tensor (All the following are equivalent)
tri.sum() # tensor(3.)
torch.sum(tri)
```

**Vector and Matrix Operations**

```python
# Vector Dot Product (Between tensors a and b. Ensure proper dimensions)
# You can equivalently use .inner() for higher dimensional inner products
torch.dot(a, b)
a.dot(b)

# Matrix Multiplication (Between tensors a and b. Ensure proper dimensions)
a @ b
torch.mm(a, b)
torch.matmul(a, b)
a.matmul(b)
```



### Conditions

You can do a lot of cool stuff with conditions!

```python
row = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 8, 9])
# tensor([0., 1., 2., 3., 4., 5., 6., 8., 9.])

square = row.reshape(3, 3)
# tensor([[0., 1., 2.],
#         [3., 4., 5.],
#         [6., 8., 9.]])

# Get condition tensor
square > 3
# tensor([[False, False, False],
#         [False,  True,  True],
#         [ True,  True,  True]])

# Now we can index into the tensor using the condition tensor!!
square[square > 3] # tensor([4., 5., 6., 8., 9.])
```



### Where()

[Docs](https://pytorch.org/docs/stable/generated/torch.where.html)

Two uses. One to get indices, and another to get altered tensors.

> **Note**: `torch.where()` is different from `Tensor.where()`! Their signatures are similar but behaviour can differ greatly.

**To Get Indices**

```python
# square:
# tensor([[0., 1., 2.],
#         [3., 4., 5.],
#         [6., 8., 9.]])

square.where(square > 0) # (tensor([0, 0, 1, 1, 1, 2, 2, 2]), tensor([1, 2, 0, 1, 2, 0, 1, 2]))

square[torch.where(square > 0)] # tensor([1., 2., 3., 4., 5., 6., 8., 9.])
square[square.nonzero(as_tuple=True)] # Equivalent shorthand
```

> **Note**: `torch.where()` is different from `square.where()`!

**To Get Altered Tensors**

> Return a tensor of elements selected from either `x` or `y`, depending on `condition`.
>
> The operation is defined as:
>
> ![image-20210617031626985](assets/01%20PyTorch%20Basics/image-20210617031626985.png)
>
> [Source](https://pytorch.org/docs/stable/generated/torch.where.html)

```python
y = torch.ones(3, 2)
x = torch.randn(3, 2)
# tensor([[-0.4620,  0.3139],
#         [ 0.3898, -0.7197],
#         [ 0.0478, -0.1657]])

torch.where(x > 0, x, y) # If condition, x, else y
# tensor([[ 1.0000,  0.3139],
#         [ 0.3898,  1.0000],
#         [ 0.0478,  1.0000]])
```

If you want to use `Tensor.where()`, you can use this alternate use. But again, **take note that the signature is different from** `torch.where()`!

```python
# This one uses broadcasting
square.where(square > 5, torch.Tensor([0.]))
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [6., 8., 9.]])
```



### NumPy Bridge

> **Note**: This only works for CPU tensors! If you're working with GPU tensors, convert them to CPU first using `Tensor.cpu()`

When you get the numpy representation of a torch array, it **shares memory locations**! So any ops done on the array in numpy **is reflected in the torch Tensor** and vice versa!

Use `.numpy()` to get the numpy representation. And `.from_numpy()` to go from a numpy array to a Tensor.

```python
# To numpy
a = torch.ones(6)
b = a.numpy()

# From numpy
a = np.ones(6)
b = torch.from_numpy(a)

# From more general sources (including NumPy)
torch.as_tensor(a) # Slightly higher overhead, usually does not copy
```



## Building Modules

Hooray, now we can finally get to learning about learning!



### Concepts

#### **Computation Graphs**

Computation graphs define sequences of operations going from input to model output. Graph edges represent tensors going in and out, and the nodes in the graph represent operations.

**Example computation graph** of linear regression: ![image-20210617043326438](assets/01%20PyTorch%20Basics/image-20210617043326438.png)

![Linear Regression Computation Graph](assets/01%20PyTorch%20Basics/IcBhTjS.png)

PyTorch lets you specify **arbitrary computation graphs**!



#### **Gradient Descent Learning**

<img src="assets/01%20PyTorch%20Basics/gradient_descent_line_graph.gif" alt="Intro to Gradient Descent | Fewer Lacunae" width="80%" />

[Image Source](https://kevinbinz.com/2019/05/26/intro-gradient-descent/)

If you can compute the **partial derivative** of each model parameter, you can do gradient descent to optimise the performance of your model! You do this by literally **update your model parameters by following the gradient**, at some learning rate for N epochs.

[Key Concepts](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/):

- Epoch: How many training iterations to do
  - Usually you will want to randomly sample from your training set some number of times per epoch
- Batch: How many samples to work through before updating your model parameters
- Learning Rate: How much to adjust your model parameters (scaled by gradient)

An epoch may be comprised of one or more batches.



### AutoGrad Basics

> `torch.autograd` is PyTorch’s **automatic differentiation engine** that powers neural network training.
>
> [Source](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

#### **Tracking Gradients**

In order to do gradient descent, you need to track gradients! Let's create a tensor that tracks its gradient for example purposes!

> **Note**: Most times you shouldn't be doing this unless you explicitly want a gradient. Since in a proper neural net (NN) program, the model parameters will handle it for you. But in this case, for explanation purposes we need to track gradients explicitly.
>
> **Note**: Also, take note that here `.tensor()` is used, which is **different** from `.Tensor()`!!

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)

# You can do some ops also
x.data # Get underlying data
x.grad_fn # Access where the tensor came from
x.requires_grad # True

# Some additional stuff
tri = torch.Tensor([0, 1, 2])
tri.requires_grad_() # Explicitly start tracking gradient now
tri.detach_() # Stop tracking the gradient
```

Let's look at an example:

```python
y = torch.tensor([1., 2., 3.])

x = torch.tensor([1., 2., 3.], requires_grad=True)
x.grad_fn # None

z = x + y
z # tensor([2., 4., 6.], grad_fn=<AddBackward0>)
```

Cool! We can see that `x`'s `grad_fn` is None because it was user created, and that `z` tracked where it came from in its own `grad_fn` (from the addition operation.)



#### **Computing Gradients**

Ok, now that we know how to track gradients, let's actually compute backpropagated gradients!

- Call `backward()` to backprop from the tensor you're calling from up the computation path until it reaches the input
- Access `.grad` on the **input** to get the gradient after backprop

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = torch.tensor([4., 5., 6.], requires_grad=True)
z = x + y

z_sum = torch.sum(z)
z_sum.backward()
print(x.grad) # tensor([1., 1., 1.])

# Now, if we called .backward() again the gradient keeps increasing!
z_sum.backward()
print(x.grad) # tensor([2., 2., 2.])
```

The gradient accumulates with each call of `backward()`. Generally we'll want to explicitly zero out gradients before calling `backward()` to run backpropagation.

> **Zeroing Gradients**
>
> ```python
> # If working with raw tensors, access the grad member directly
> .grad.zero_()
> 
> # Otherwise, if working with parameters or optimizers
> optimizer.zero_grad()
> ```



### Building Your Own Model

#### **Module**

You can build your own model by subclassing `torch.nn.Module`! Override the `__init__()` and `forward()` methods.

> **Note**: Remember to call the `torch.nn.Module` constructor in your overridden `__init__`
>
> ```python
> super(YourClass, self).__init__()
> ```

> - In `__init__()`, you should take arguments that modify how the model runs (e.g. # of layers, # of hidden units, output sizes). You'll also set up most of the layers that you use in the forward pass here.
> - In `forward()`, you define the "forward pass" of your model, or the operations needed to transform input to output. **You can use any of the Tensor operations in the forward pass.**
>
> [Source](https://colab.research.google.com/drive/1HhLqlpYr6ZUT6u-PDWq5L73muY4JvnpJ?usp=sharing#scrollTo=RZNnUJcrU2r2)

These modules then can be used as a node on the computation graph! You can **call them** as an operation, and they'll apply their `forward()` method to the input tensor.



#### **Dataset Splitting**

You can do manual slicing...

```python
test = torch.cuda.FloatTensor(pd.read_csv(test_path).to_numpy())
train = torch.cuda.FloatTensor(pd.read_csv(train_path).to_numpy())

train_x, train_y = train[:, :-1], train[:, -1:]
```

Or use the PyTorch utility

```python
torch.utils.data.random_split(dataset, lengths)

# Example
train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
```



#### **Training: Gradient Descent**

Once you have your model, you can run gradient descent! Here, we'll use stochastic gradient descent (SGD)

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_function = nn.BCELoss() 

for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```



### Example: Logistic Regression

[Source](https://colab.research.google.com/drive/1HhLqlpYr6ZUT6u-PDWq5L73muY4JvnpJ?usp=sharing#scrollTo=RZNnUJcrU2r2)

Let's try making our own model for logistic regression!

```python
# Modified from SUTD

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

# MODEL =======================================================================
class LogisticRegression(nn.Module):
  def __init__(self, input_size, num_classes):
    # Always call the superclass (nn.Module) constructor first!
    super(LogisticRegression, self).__init__()
    self.linear = nn.Linear(input_size, num_classes) # (in_features, out_features)

  def forward(self, x):
    out = self.linear(x)
    out = torch.sigmoid(out) # Softmax for log-probability
    return out # Shape: (batch_size, num_classes)


# Now we can create and call it ===============================================
logreg_clf = LogisticRegression(input_size=2, num_classes=1)
# LogisticRegression(
#   (linear): Linear(in_features=2, out_features=1, bias=True)
# )

logreg_clf(torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]]))
# tensor([[0.4476],
#         [0.4385]], grad_fn=<SigmoidBackward>)


# Set up training ==============================================================
lr_rate = 0.001

# Get training set of input X and labels Y
X = torch.Tensor([[0,0],[0,1], [1,0], [1,1]])
Y = torch.Tensor([0,1,1,0]).view(-1,1)

# Create loss function and training function
loss_function = nn.BCELoss() 
optimizer = torch.optim.SGD(logreg_clf.parameters(),
                            lr=lr_rate) # Stochastic Gradient Descent


# TRAINING LOOP ================================================================
epochs = 2001
steps = X.size(0) # 4

for i in range(epochs):
    for j in range(steps):
        data_point = np.random.randint(X.size(0)) # Random Sampling

        x_var = torch.Tensor(X[data_point]) # Input
        y_var = torch.Tensor(Y[data_point]) # Label

        optimizer.zero_grad() # Zero the gradients
        y_hat = logreg_clf(x_var) # Forward pass

        loss = loss_function(y_hat, y_var) # Calculate loss
        loss.backward() # Backprop
        optimizer.step() # Update Parameters

    if i % 500 == 0: # For progress reports
        print ("Epoch: {0}, Loss: {1}, ".format(i, loss.data.numpy()))


# APPLY MODEL AFTER TRAINING ===================================================
# In this case, our data actually was for a XOR operation (:
test = [[0,0], [0,1], [1,1], [1,0]]

for trial in test: 
    Xtest = torch.Tensor(trial)
    y_hat = logreg_clf(Xtest) # Predict
  
    prediction = y_hat > 0.5
    print("{0} xor {1} = {2}".format(int(Xtest[0]), int(Xtest[1]), prediction))
    
    # 0 xor 0 = 1 (Wrong)
    # 0 xor 1 = 0 (Wrong)
    # 1 xor 1 = 0 (Correct)
    # 1 xor 0 = 1 (Correct)
```

```python
# Out:
Epoch: 0, Loss: 0.7317752242088318, 
Epoch: 500, Loss: 0.7088961601257324, 
Epoch: 1000, Loss: 0.6454429030418396, 
Epoch: 1500, Loss: 0.7373309135437012, 
Epoch: 2000, Loss: 0.6999746561050415, 
```

> **Notes**
>
> Notice in this case our batch size is 1. And our epochs don't necessarily sample the entire training set each time (due to the random sampling).
>
> Also, our loss remains high and the model performs horribly! This is because we actually need a non-linear model (the data and labels are actually for a XOR function, which is non-linear). But it's fine.



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

![image-20210630005207722](assets/01%20PyTorch%20Basics/image-20210630005207722.png)

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

![What is ReLU and Softmax? - Quora](assets/01%20PyTorch%20Basics/main-qimg-07bc0ec05532caf5ebe8b4c82d0f5ca3)

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

![image-20210627041903528](assets/01%20PyTorch%20Basics/image-20210627041903528.png)



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
>![image-20210630020010846](assets/01%20PyTorch%20Basics/image-20210630020010846.png)
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

> ![Understanding hidden memories of recurrent neural networks | the morning  paper](assets/01%20PyTorch%20Basics/rnnvis-fig-2.jpeg)
>
> [Source](https://blog.acolyer.org/2019/02/25/understanding-hidden-memories-of-recurrent-neural-networks/)

```python
# Where hidden_size is the number of features of the hidden state h
rnn = nn.RNN(input_size, hidden_size, out_size)
```



#### **LSTM**

[Docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

> ![Pytorch中nn.LSTM与nn.LSTMCell_XiaobaiLaplace的博客-CSDN博客_nn.lstmcell](assets/01%20PyTorch%20Basics/20201120150748488.png)
>
> [Image Source](https://blog.csdn.net/XiaobaiLaplace/article/details/109842576)
>
> Where h is the hidden state, c is the cell state.

- input_size: Number of features in input

- hidden_size: Number of features in the hidden state

- num_layers: Number of recurrent layers (e.g. num_layers=2 is two LSTM units, with the first feeding its output to the second.)

  - > num_layers example with 2 layers
    >
    > ![LSTM的num_layers是什么意思？ 已解决- MXNet Gluon - MXNet / Gluon 论坛](assets/01%20PyTorch%20Basics/21fd219f0964b5981f5534eac93fa30ab9be3460.jpg)
    >
    > [Image Source](![LSTM的num_layers是什么意思？ 已解决- MXNet Gluon - MXNet / Gluon 论坛](assets/01%20PyTorch%20Basics/21fd219f0964b5981f5534eac93fa30ab9be3460.jpg))

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

