# PyTorch Introduction (Tensors and Modules)

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

> If you already know what's going on, you can just look at the resources directory instead (there are some examples!)



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

<img src="assets/01%20PyTorch%20-%20Introduction%20(Tensors%20and%20Modules)/tensor-examples.jpg" alt="PyTorch Tensor Basics - KDnuggets" width="50%" />

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

![array](assets/01%20PyTorch%20-%20Introduction%20(Tensors%20and%20Modules)/array.jpg)

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
> ![image-20210617031626985](assets/01%20PyTorch%20-%20Introduction%20(Tensors%20and%20Modules)/image-20210617031626985.png)
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



## Modules and Autograd

Hooray, now we can finally get to learning about learning!



### Concepts

#### **Computation Graphs**

Computation graphs define sequences of operations going from input to model output. Graph edges represent tensors going in and out, and the nodes in the graph represent operations.

**Example computation graph** of linear regression: ![image-20210617043326438](assets/01%20PyTorch%20-%20Introduction%20(Tensors%20and%20Modules)/image-20210617043326438.png)

![Linear Regression Computation Graph](assets/01%20PyTorch%20-%20Introduction%20(Tensors%20and%20Modules)/IcBhTjS.png)

PyTorch lets you specify **arbitrary computation graphs**!



#### **Gradient Descent Learning**

<img src="assets/01%20PyTorch%20-%20Introduction%20(Tensors%20and%20Modules)/gradient_descent_line_graph.gif" alt="Intro to Gradient Descent | Fewer Lacunae" width="80%" />

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

> **Alternatively**, if you have a PyTorch `Dataset`, you can use `Subset`
>
> ```python
> # Source: https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987/5
> 
> from torch.utils.data import Subset
> from sklearn.model_selection import train_test_split
> 
> def train_val_dataset(dataset, val_split=0.25):
>     train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
>     datasets = {}
>     datasets['train'] = Subset(dataset, train_idx)
>     datasets['val'] = Subset(dataset, val_idx)
>     return datasets
> ```



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



## Utilities

### Saving and Loading Models

```python
# Save
torch.save(model.state_dict(), "model_name.pth")

# Load
model.load_state_dict(torch.load("model_name.pth"))
```



### Datasets

[Official Dataset and Dataloader Tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

You can download or create your own **datasets** to iterate through! Then you can use those datasets in a **dataloader** that helps you manage batching, and even helps you manage distributed training across multiple processes!



#### **Download a Pre-Existing Dataset**

[See all PyTorch datasets here](https://pytorch.org/vision/stable/datasets.html)

> In this case, a single **transform** `ToTensor` is used, which causes all dataset elements taken from the dataset to be subjected to that transform in a pre-processing step as each element is iterated through.
>
> You can very easily specify a chain of transforms to be used instead, using `transforms.Compose`.

```python
from torchvision import datasets
from torchvision.transforms import ToTensor

# Load training data
training_data = datasets.FashionMNIST(
    root="data", # Specify download folder
    train=True,
    download=True,
    transform=ToTensor()
)

# Load without training data
test_data = datasets.FashionMNIST(
    root="data", # Specify download folder
    train=False,
    download=True,
    transform=ToTensor()
)
```



#### **Load a Pre-Existing Dataset**

Alternatively, if you have a folder of images somewhere, you can load the data using an `ImageFolder` instance, running pre-processing using `torchvision.transforms`!

```python
train_dataset = datasets.ImageFolder("train-uva", train_transform)
val_dataset = datasets.ImageFolder("val-uva", test_transform)
```



#### **Iterate Through Dataset**

You can just index through a dataset manually like a list! You'll get the data **and** the label for each index!

```python
img, label = training_data[index]

# You can also do random sampling
sample_idx = torch.randint(len(training_data), size=(1,)).item()
img, label = training_data[sample_idx]
```



#### **Build a Custom Dataset**

Finally, if you have very specific needs, you can write your own dataset class, just subclass from `torch.utils.data.Dataset`

Here's an example from the [tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

> Note: In this case, we're creating a map-style dataset, suitable for when random accesses are cheap.
>
> However, if you need to have an iterable-style dataset because random accesses are expensive, you need to subclass [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) instead!

```python
# Source: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
from torch.utils.data import Dataset

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        # Run at the start
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):  # For handling a len() call
        return len(self.img_labels)

    def __getitem__(self, idx):  # For handling iteration and indexing
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]  # Index into image labels (since it is a pandas dataframe)

        if self.transform:  # Only transform on load!
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```



### DataLoaders

[Docs](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) | [Official Dataset and DataLoader Tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

Once you have a dataset, you can load it in a **dataloader** that helps you manage batching, and even helps you manage distributed training across multiple processes!



#### **Initialise and Use DataLoader**

You can even specify whether you need your examples shuffled, and the batch size!

Then, for each iteration of the DataLoader, it'll load as many elements as specified in batch_size!

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

Then, you can just iterate through the dataloader!

```python
train_features, train_labels = next(iter(train_dataloader))

# Or using a for loop
for train_features, train_labels in train_dataloader:
    # Do something
```

You can change how you get your train elements by changing the sampler, batch sampler, or collate function (which takes batches and processes/reshapes them into a usable form for neural network input)! [See a tutorial](https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html), or [another one](https://medium.com/geekculture/pytorch-datasets-dataloader-samplers-and-the-collat-fn-bbfc7c527cf1).

- Samplers generate a sequence of indices for the whole dataset
- Batch samplers wrap samplers or iterables of sequences to yield a mini-batch of indices
- Collate functions 'collate' a batch (hard to explain, you should read the docs for a more thorough treatment)



#### **Distributed Training**

[See blog post](https://towardsdatascience.com/this-is-hogwild-7cc80cd9b944)

But in short, you set your model to `.share_memory()`, and use a `DistributedSampler` in your `DataLoader `.

> **Note**: Using `IterableDataset` throws a wrench into multiprocessing. Special care has to be taken to split the dataset across each worker instance. See the [docs](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) for more information. Unfortunately, you can't use a sampler with iterable datasets!

> **Note**: It is generally not recommended to return CUDA tensors in multi-process loading because of many subtleties in using CUDA and sharing CUDA tensors in multiprocessing (see [CUDA in multiprocessing](https://pytorch.org/docs/stable/notes/multiprocessing.html#multiprocessing-cuda-note)). Instead, we recommend using [automatic memory pinning](https://pytorch.org/docs/stable/data.html#memory-pinning) (i.e., setting `pin_memory=True`), which enables fast data transfer to CUDA-enabled GPUs.
>
> [Docs](https://pytorch.org/docs/stable/data.html#map-style-datasets)



### Plotting Loss Evolution

This function handily keeps track of entered epochs and losses internally!

```python
import matplotlib.pyplot as plt

def record_losses(epoch=None, loss=None,
                  vis=True, vis_only=False,
                  clear=False, clear_only=False,
                  _memory={'epochs': [], 'losses': []}):
    epochs, losses = _memory['epochs'], _memory['losses']

    if clear or clear_only:
        epochs.clear()
        losses.clear()

        if clear_only:
            return _memory

    if not vis_only:
        epochs.append(epoch)
        losses.append(loss)

    if vis:
        plt.plot(epochs, losses, label="Loss")

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Evolution")
        plt.legend()
        plt.show()

    return _memory
```

Call it like so:

```python
record_losses(<epoch>, <loss>, vis=False) # Append without visualising
record_losses(<epoch>, <loss>) # Visualise loss evolution

record_losses(<epoch>, <loss>, clear=True) # To reset the internal memory

# DO NOT TOUCH THE _memory variable!!
```



### Detect Loss Convergence

Good for detecting convergence!

```python
def converged(losses, window=10, threshold=0.0001):
    try:
        if len(losses) < window:
            return False
        
        losses = losses[-window:]
        return max(losses) - min(losses) < threshold
    except:
        return False
```

