# Generic utilities for training models
# From SUTD

from torch.utils.data import DataLoader

def train_model(model,
                loss_fn,
                batchSize,
                trainset,
                valset,
                optimizer,
                num_epochs,
                device="cuda",
                log=True):
    """Generic training function, with minibatch."""
    # Shuffling is needed in case dataset is not shuffled by default.
    train_loader = DataLoader(dataset = trainset, batch_size = batchSize,
                              shuffle = True)

    # We don't need to bach the validation set but let's do it anyway.
    val_loader = DataLoader(dataset = valset, batch_size = batchSize,
                            shuffle = False) # No need.

    # Accuracy and loss loggers.
    train_accuracies = []; val_accuracies = []
    train_losses = []; val_losses = []

    # GPU enabling.
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    for epoch in range(0, num_epochs):
        correct, cum_loss = 0.0, 0.0
        model.train() # Set model to train mode

        # Training pass
        for (i, (inputs, labels)) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            scores = model(inputs) # Forward pass
            loss = loss_fn(scores, labels)

            # Count how many correct in this batch.
            max_scores, max_labels = scores.max(1)
            correct += (max_labels == labels).sum().item()
            cum_loss += loss.item()

            optimizer.zero_grad()
            loss.backward() # Backprop
            optimizer.step()

            # Logging current training results
            if (i + 1) % 100 == 0 and log:
                print('Train-epoch %d. Iteration %05d / %05d, Avg-Loss: %.4f, Accuracy: %.4f' %
                      (epoch, i + 1, len(train_loader), cum_loss / (i + 1), correct / ((i + 1) * batchSize)))

        train_accuracies.append(correct / len(trainset))
        train_losses.append(cum_loss / (i + 1))

        # Validation pass
        correct, cum_loss = 0.0, 0.0
        model.eval()
        for (i, (inputs, labels)) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            scores = model(inputs) # Forward
            cum_loss += loss_fn(scores, labels).item()

             # Count how many correct in this batch.
            max_scores, max_labels = scores.max(1)
            correct += (max_labels == labels).sum().item()

        val_accuracies.append(correct / len(valset))
        val_losses.append(cum_loss / (i + 1))

        # Logging current validation results
        if log:
            print(
                'Validation-epoch %d. Avg-Loss: %.4f, Accuracy: %.4f'
                % (epoch, cum_loss / (i + 1), correct / len(valset))
            )
