import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_training_curve(catboost_info_path, loss_function, metrics, epoch):
    train_tsv = os.path.join(catboost_info_path, "learn_error.tsv")
    val_tsv = os.path.join(catboost_info_path, "test_error.tsv")
    with open(train_tsv, 'r') as f:
        train_info = pd.read_csv(f, sep="\t")
    with open(val_tsv, 'r') as f:
        val_info = pd.read_csv(f, sep="\t")
    fig, ax = plt.subplots()
    ax.plot(list(range(epoch)), train_info[loss_function], label="train loss", color='blue')
    ax.plot(list(range(epoch)), val_info[loss_function], label="val loss", color='blue', linestyle='dashed')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Function")
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.2))
    ax2 = ax.twinx()
    ax2.plot(list(range(epoch)), train_info[metrics], label="train acc", color='red')
    ax2.plot(list(range(epoch)), val_info[metrics], label="val acc", color='red', linestyle='dashed')
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc='center left', bbox_to_anchor=(1.1, 0.8))
    plt.savefig(os.path.join(catboost_info_path, "training_curve.png"), bbox_inches='tight')
    plt.show()

