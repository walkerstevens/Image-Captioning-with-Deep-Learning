import matplotlib.pyplot as plt
import uuid

# ----------------------------------------------------------------------------------------------------------------------
# NOTE !!!
# - All numbers below were extracted from the stdout of the training using shell commands.
# - Training output is also committed in this repo in .txt format
# ----------------------------------------------------------------------------------------------------------------------

loss = {
    # training
    'T': [3.9201, 3.4699, 3.3448, 3.2123, 3.1675, 3.1514, 3.1316, 3.1316, 3.0728, 3.0493],
    # validation
    'V': [3.471, 3.346, 3.298, 3.274, 3.259, 3.255, 3.252, 3.250, 3.251, 3.254],

}
top_5_acc = {
    # training
    'T': [66.425, 67.893, 72.088, 73.696, 75.443, 76.042, 76.522, 76.951, 77.309, 77.643],
    # validation
    'V': [72.013, 73.636, 74.348, 74.846, 74.896, 74.927, 75.007, 75.100, 75.123, 75.135]
}
bleu_4 = {
    # validation
    'V': [0.1854003933845439, 0.20227172639889843, 0.20599541336395014, 0.21568956467433501, 0.21068791198537162, 0.21284994339925392, 0.21322736090053196, 0.21214215843927964, 0.21550289452494514, 0.21472683187692296]
}

# points to the exact same parameters in train.py
epochs = 10
batch_size = 64
decoder_lr = 4e-4
experiment_title = "epochs: {} | batch_size: {} | decoder_lr = {}".format(epochs, batch_size, decoder_lr)


def plot_loss():
    plt.plot(loss['T'], label="Train.")
    plt.plot(loss['V'], label="Valid.")
    plt.legend(loc="center right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks([i for i in range(9)])
    plt.figtext(0.5, 0.9, experiment_title, ha='center', va='center')

    plt.savefig('plots/learning_curves/loss-{}.png'.format(str(uuid.uuid4())))
    plt.show()


def plot_accuracy():
    plt.plot(top_5_acc['T'], label="Train.")
    plt.plot(top_5_acc['V'], label="Valid.")
    plt.legend(loc="center right")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks([i for i in range(9)])
    plt.figtext(0.5, 0.9, experiment_title, ha='center', va='center')

    plt.savefig('plots/learning_curves/accuracy-{}.png'.format(str(uuid.uuid4())))
    plt.show()


def plot_bleu_score():
    plt.plot(bleu_4['V'], label="Train.")
    plt.legend(loc="center right")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU-4")
    plt.xticks([i for i in range(9)])
    plt.figtext(0.5, 0.9, experiment_title, ha='center', va='center')

    plt.savefig('plots/learning_curves/bleu4-{}.png'.format(str(uuid.uuid4())))
    plt.show()


if __name__ == '__main__':
    plot_loss()
    plot_accuracy()
    plot_bleu_score()