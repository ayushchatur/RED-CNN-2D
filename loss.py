# import numpy as np
# import matplotlib.pyplot as plt

# # Load the losses from the file
# losses = np.load('save/train_losses.npy')
# # print(losses)
# # Plot the losses
# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Losses')
# plt.savefig('training_losses_plot.png')
import numpy as np
import matplotlib.pyplot as plt
samples = 500
# Load the losses from the file
losses = np.load('save2/train_losses.npy')
# print(losses)
# print(len(losses))

# Subsample the losses to plot every 100th value
subsampled_losses = losses[::samples]

# Generate x-axis values for subsampled data
epochs = np.arange(0, len(losses), samples)

# Plot the subsampled losses
plt.plot(epochs, subsampled_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses (Every 500 Epoch)')

plt.savefig('loss_curve_epochs200.png')

