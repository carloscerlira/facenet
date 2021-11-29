import matplotlib.pyplot as plt 
plt.style.use("seaborn")
from sklearn import metrics
import numpy as np 
import pandas as pd 
import pickle 

def get_metrics(y_true, y_pred, threshold=1.3):
    tp = tn = fp = fn = 0
    for yt, yp in zip(y_true, y_pred):
        is_same = yp <= threshold
        if(yt == 1):
            tp += is_same 
            fn += (is_same == 0)
        if(yt == 0):
            tn += (is_same == 0)
            fp += is_same  

    fpr = fp/(fp+tn)
    tpr = tp/(tp+fn)
    acc = (tp+tn)/(tp+tn+fp+fn)
    return fpr, tpr, acc 

def gen_roc():
    phases = ['train', 'test']
    for phase in phases:
        filename = f'./logs/scores/scores_{phase}.obj'
        infile = open(filename,'rb')
        scores = pickle.load(infile)
        infile.close()

        y_true, y_pred = scores 
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        
        x = np.linspace(0, 1, 10)
        plt.plot(x, x, linestyle='--', color="black", alpha=0.5)

        auc = metrics.auc(tpr, fpr)
        plt.plot(tpr, fpr, label=f"{phase}, AUC: {auc:.3f}")

    plt.legend()
    plt.title("ROC Curve")
    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")
    plt.savefig("./logs/img/roc.png")
    plt.close()

def gen_loss_curve():
    df = pd.read_csv("./logs/scores/loss.csv")
    df["loss"].plot()
    plt.title("Loss function for each epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss function")
    plt.savefig("./logs/img/loss.png")
    plt.close()

# df = pd.read_csv("stats_s1.csv")
# plt.title("Accuracy for each epoc")
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# df["accuracy"].plot()
# plt.savefig("./logs/img/accuracy.png")
# plt.close()
# # plt.show()