import pandas as pd
import matplotlib.pyplot as plt


def plot(path):
    data = pd.read_csv(path, delimiter=';')
    epoch = data['epoch']
    acc = data['acc'] * 100
    loss = data['loss']
    val_acc = data['val_acc'] * 100
    val_loss = data['val_loss']
    plt.plot(epoch, acc, label='Training accuracy')
    plt.plot(epoch, val_acc, label='Validation accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.show()
    plt.plot(epoch, loss, label='Training loss')
    plt.plot(epoch, val_loss, label='Validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


plot('Logs/ConvNet_1_1.csv')
