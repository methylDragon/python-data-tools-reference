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
rand_float = torch.rand(4, 6) # Random between 0.0 and 1.0
ones_float = torch.ones(4, 6)
zeros_float = torch.zeros(4, 6)

# From other tensor's dimensions
x_ones = torch.ones_like(x_data) # Retains the properties of x_data, but all ones
x_rand = torch.rand_like(x_data, dtype=torch.float) # Overrides the datatype of x_data
```



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

**Matrix Operations**

```python
# Matrix Dot Product (Between tensors a and b. Ensure proper dimensions)
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



## Learning

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

You can build your own model by subclassing `torch.nn.Module`! Override the `__init__()` and `forward()` methods.

> **Note**: Remember to call the `torch.nn.Module` constructor in your overridden `__init__`
>
> ```python
> super(LogisticRegression, self).__init__()
> ```

> - In `__init__()`, you should take arguments that modify how the model runs (e.g. # of layers, # of hidden units, output sizes). You'll also set up most of the layers that you use in the forward pass here.
> - In `forward()`, you define the "forward pass" of your model, or the operations needed to transform input to output. **You can use any of the Tensor operations in the forward pass.**
>
> [Source](https://colab.research.google.com/drive/1HhLqlpYr6ZUT6u-PDWq5L73muY4JvnpJ?usp=sharing#scrollTo=RZNnUJcrU2r2)

These modules then can be used as a node on the computation graph! You can **call them** as an operation, and they'll apply their `forward()` method to the input tensor.



### Example: Logistic Regression

[Source](https://colab.research.google.com/drive/1HhLqlpYr6ZUT6u-PDWq5L73muY4JvnpJ?usp=sharing#scrollTo=RZNnUJcrU2r2)

Let's try making our own model for logistic regression!

```python
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

**Notes**

Notice in this case our batch size is 1. And our epochs don't necessarily sample the entire training set each time (due to the random sampling).

Also, our loss remains high and the model performs horribly! This is because we actually need a non-linear model (the data and labels are actually for a XOR function, which is non-linear). But it's fine.

```
Epoch: 0, Loss: 0.7317752242088318, 
Epoch: 500, Loss: 0.7088961601257324, 
Epoch: 1000, Loss: 0.6454429030418396, 
Epoch: 1500, Loss: 0.7373309135437012, 
Epoch: 2000, Loss: 0.6999746561050415, 
```

