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


def converged(losses, window=10, threshold=0.0001):
    """"Detect loss convergence in a window."""
    try:
        if len(losses) < window:
            return False

        losses = losses[-window:]
        return max(losses) - min(losses) < threshold
    except:
        return False
