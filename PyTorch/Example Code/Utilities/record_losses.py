import matplotlib.pyplot as plt

def record_losses(epoch, loss,
                  vis=True, clear=False,
                  _memory={'epochs': [], 'losses': []}):
    epochs, losses = _memory['epochs'], _memory['losses']

    if clear:
        epochs.clear()
        losses.clear()

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
