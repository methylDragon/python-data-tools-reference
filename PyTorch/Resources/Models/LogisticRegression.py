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

if __name__ == "__main__":
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
