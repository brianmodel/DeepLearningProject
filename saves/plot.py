import matplotlib.pyplot as plt
import json
from matplotlib.ticker import MaxNLocator

with open('/home/gigachad/DeepLearningProject/saves/run-.-tag-Training accuracy.json', 'r') as f:
    train_acc = json.load(f)

with open('/home/gigachad/DeepLearningProject/saves/run-.-tag-Training loss.json', 'r') as f:
    train_loss = json.load(f)

with open('/home/gigachad/DeepLearningProject/saves/run-.-tag-Validation accuracy.json', 'r') as f:
    val_acc = json.load(f)

with open('/home/gigachad/DeepLearningProject/saves/run-.-tag-Validation loss.json', 'r') as f:
    val_loss = json.load(f)

fig, axs = plt.subplots(1, 2)

axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[0].plot([x for _, x, _ in train_acc], [y for _, _, y in train_acc], label='Train')
axs[0].plot([x for _, x, _ in train_acc], [y for _, _, y in val_acc], label='Validation')
axs[0].legend()
axs[0].set_ylabel("Accuracy")
axs[0].set_xlabel("Epoch")
axs[0].set_title("Accuracy")

axs[1].plot([x for _, x, _ in train_loss], [y for _, _, y in train_loss])
axs[1].plot([x for _, x, _ in train_loss], [y for _, _, y in val_loss])
# axs[1].legend()
axs[1].set_ylabel("Loss")
axs[1].set_xlabel("Epoch")
axs[1].set_title("Loss")
plt.tight_layout()
# fig.legend()
plt.show()