import matplotlib.pyplot as plt
import os
import pandas as pd
import plotly.graph_objects as go

from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize


def plot_training_curve(catboost_info_path, loss_function, metrics, epoch):
    Path("output").mkdir(exist_ok=True)
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
    plt.savefig("output/training_curve.png", bbox_inches='tight')
    plt.show()


def roc(y_true, y_pred, class_list, show_plots=False):
    """
    This method plots roc curve and return roc auc score for each class
    :param y_true: ground truth label
    :type y_true: a list or numpy array
    :param y_pred: predicted probabilities of shape: n x m (n is the number of data samples, m is the number of classes)
    :type y_pred: 2d list or 2d numpy array
    :param class_list: all classes list, in the same order with y_pred axis 1
    :type class_list: list
    :param show_plots: if show single roc curve plot for each class
    :type show_plots: boolean
    :return: auc score of each class
    :rtype: dictionary
    """
    Path("model_analysis_result/roc_curve/").mkdir(parents=True, exist_ok=True)
    y_test = label_binarize(y_true, classes=class_list)  # encode to one-hot vector
    n_classes = len(class_list)
    auc_dict = {}
    fpr_tpr = {}
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    # plot all curve in one plot
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        fpr_tpr[class_list[i]] = (fpr, tpr)
        auc_dict[class_list[i]] = roc_auc
        plt.plot(
            fpr,
            tpr,
            label="{} AUC score ={:.3f})".format(class_list[i], roc_auc))

    plt.title("Total ROC curve")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('False Positive Rate\n Specificity')
    plt.ylabel('True Positive Rate\n Recall\n Sensitivity')
    plt.show()
    plt.savefig("model_analysis_result/roc_curve/roc_curve_total.png", bbox_inches='tight')
    # plot and save roc curve for each class
    for i in range(n_classes):
        class_ = class_list[i]
        fpr, tpr = fpr_tpr[class_]
        auc_score = auc_dict[class_]
        plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
        plt.plot(
            fpr,
            tpr,
            label="AUC score ={:.3f})".format(auc_score))
        plt.title("{}".format(class_))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate\n Recall\n Sensitivity')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig("model_analysis_result/roc_curve/roc_curv_{}.png".format(class_), bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
    return auc_dict


def pr(y_true, y_pred, class_list, show_plots=False):
    """
    This method plots precision recall (PR) curve and return PR auc score for each class
    :param y_true: ground truth label
    :type y_true: a list or numpy array
    :param y_pred: predicted probabilities of shape: n x m (n is the number of data samples, m is the number of classes)
    :type y_pred: 2d list or 2d numpy array
    :param class_list: all classes list, in the same order with y_pred axis 1
    :type class_list: list
    :param show_plots: if show single roc curve plot for each class
    :type show_plots: boolean
    :return: auc score of each class
    :rtype: dictionary
    """
    Path("model_analysis_result/pr_curve/").mkdir(parents=True, exist_ok=True)
    y_test = label_binarize(y_true, classes=class_list)  # encode to one-hot vector
    n_classes = len(class_list)
    auc_dict = {}
    precision_recall = {}
    # plot all curve in one plot
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
        pr_auc = auc(recall, precision)
        precision_recall[class_list[i]] = (precision, recall)
        auc_dict[class_list[i]] = pr_auc
        plt.plot(
            recall,
            precision,
            label="{} AUC score ={:.3f})".format(class_list[i], pr_auc))

    plt.title("Total Precision Recall curve")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    plt.savefig("model_analysis_result/pr_curve/pr_curve_total.png", bbox_inches='tight')
    # plot and save roc curve for each class
    for i in range(n_classes):
        class_ = class_list[i]
        p, r = precision_recall[class_]
        auc_score = auc_dict[class_]
        no_skill = len(y_test[:, i][y_test[:, i] == 1]) / len(y_test[:, i])
        plt.plot(
            r,
            p,
            label="AUC score ={:.3f})".format(auc_score))
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.title("{}, {} support".format(class_, sum(y_test[:, i])))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig("model_analysis_result/pr_curve/pr_curve_{}.png".format(class_), bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
    return auc_dict


def confusion_matrix(model, X_test, y_test):
    Path("output").mkdir(exist_ok=True)
    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
    fig = plt.gcf()
    fig.set_size_inches(40, 30)
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('prediction', fontsize=40)
    plt.ylabel('true', fontsize=40)
    fig.savefig('output/confusion_matrix.png', dpi=500, bbox_inches='tight')
    plt.show()


def save_f1_report(y_test, y_pred):
    Path("output").mkdir(exist_ok=True)
    report = classification_report(y_test, y_pred, digits=3, output_dict=True)
    df = pd.DataFrame(report).transpose()
    pd.set_option("display.precision", 3)
    df.to_csv("output/f1_score.tsv", sep="\t", float_format='%.3f')


def confusion_heatmap(model, test_df):
    Path("output").mkdir(exist_ok=True)
    X_test = test_df.iloc[:, 1:]
    y_test = test_df['biome']
    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
    fig = plt.gcf()
    fig.set_size_inches(40, 30)
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Predicted lable', fontsize=40)
    plt.ylabel('True lable', fontsize=40)
    fig.savefig('output/confusion_matrix.png', dpi=500, bbox_inches='tight')
    plt.show()


def plot_f1_scatter(y_test, y_pred):
    Path("output").mkdir(exist_ok=True)

    report = classification_report(y_test, y_pred, digits=3, output_dict=True)
    f1 = pd.DataFrame(report).transpose()
    pd.set_option("display.precision", 3)

    f1.drop(f1.tail(3).index, inplace=True)
    f1.sort_values(by='f1-score', inplace=True)

    new_label = []
    for i, row in f1.iterrows():
        label = "{} ({})".format(row['biome'], int(row['support']))
        new_label.append(label)

    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=f1.precision, y=new_label,
                             mode='markers',
                             marker=dict(symbol='cross'),
                             name='precision'))
    fig.add_trace(go.Scatter(y=new_label, x=f1.recall,
                             mode='markers',
                             marker=dict(symbol='square'),
                             name='recall'))
    fig.add_trace(go.Scatter(y=new_label,
                             x=f1['f1-score'],
                             mode='markers',
                             name='f1-score',
                             marker=dict(
                                 size=8 + f1.support * 0.7,
                                 symbol='circle'),
                             )
                  )
    fig.update_xaxes(tickfont=dict(size=14))
    fig.update_yaxes(tickfont=dict(size=14))
    fig.update_layout(modebar_add="togglespikelines",
                      autosize=False,
                      width=1050,
                      height=1050,
                      paper_bgcolor="LightSteelBlue", )
    fig.update_layout(legend=dict(font=dict(family="Courier", size=20, color="black")),
                      legend_title=dict(font=dict(family="Courier", size=50, color="blue")))
    fig.update_layout(
        xaxis={'side': 'top'}, )
    fig.show()
    fig.write_html("output/f1_scatter_plot.html")
