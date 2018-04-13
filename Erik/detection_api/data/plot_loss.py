from matplotlib import pyplot as plt
import pandas as pd
import os

PATH_RPN_LOC = './losses/RPNLoss_localization_loss_mul_1.csv'
PATH_RPN_OBJ = './losses/RPNLoss_objectness_loss_mul_1.csv'
PATH_BC_LOC = './losses/BoxClassifierLoss_localization_loss_mul_1.csv'
PATH_BC_CLASS = './losses/BoxClassifierLoss_classification_loss_mul_1.csv'

def myPlot(path, name):
    plt.figure()
    data = pd.read_csv(path)
    plt.plot(data['Step'], data['Value'])
    plt.xlabel('steps')
    plt.ylabel('Loss')
    plt.title(name)
    root, ext = os.path.splitext(path)
    plt.savefig(root + '.png')


myPlot(PATH_RPN_LOC, 'RPN Localization Loss')
myPlot(PATH_RPN_OBJ, 'RPN Objectness Loss')
myPlot(PATH_BC_LOC, 'BoxClassifier Localization Loss')
myPlot(PATH_BC_CLASS, 'BoxClassifier Classification Loss')