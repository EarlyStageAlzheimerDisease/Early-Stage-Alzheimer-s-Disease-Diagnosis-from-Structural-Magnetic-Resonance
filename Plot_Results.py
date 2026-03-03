from itertools import cycle
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn import metrics


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])

No_of_Dataset = 2
def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'DMO-3DAEANet', 'LOA-3DAEANet', 'NRO-3DAEANet', 'DPOA-3DAEANet', 'It-PDPO-3DAEANet']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Dataset = [1, 2]
    for i in range(2):
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Report for ', 'Dataset-', Dataset[i],
              '--------------------------------------------------')
        print(Table)

        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='DMO-3DAEANet')
        plt.plot(length, Conv_Graph[1, :], color=[0, 0.5, 0.5], linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='LOA-3DAEANet')
        plt.plot(length, Conv_Graph[2, :], color=[0.5, 0, 0.5], linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='NRO-3DAEANet')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='DPOA-3DAEANet')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='It-PDPO-3DAEANet')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Plot_ROC_Curve():
    lw = 2
    cls = ['CNN', 'ResNet', 'Unet', 'EAN', 'It-PDPO-3DAEANet ']
    for a in range(2):  # For 2 Datasets
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')
        # Actual = np.load('Target.npy', allow_pickle=True)

        colors = cycle(["blue", "darkorange", "cornflowerblue", "deeppink", "black"])  # "aqua",
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i], )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path1)
        plt.show()


def plot_results_Batch():
    eval1 = np.load('Eval_all_Batch.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC']
    Algorithm = ['TERMS', 'DMO-3DAEANet', 'LOA-3DAEANet', 'NRO-3DAEANet', 'DPOA-3DAEANet', 'It-PDPO-3DAEANet']
    Classifier = ['TERMS', 'CNN', 'ResNet', 'Unet', 'EAN', 'It-PDPO-3DAEANet ']
    Batch = [16, 32, 64, 128, 256]
    for i in range(eval1.shape[0]):
        for m in range(eval1.shape[1]):
            value1 = eval1[i, m, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Terms[:5])
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value1[j, :5])
            print('-------------------------------------------------- Dataset -', str(i + 1), 'Batch Size-'+ str(Batch[m])+
                  ' -  Method Comparison',
                  '--------------------------------------------------')
            print(Table)


            value2 = eval1[i, m, :, 4:]
            Table = PrettyTable()
            Table.add_column(Classifier[0], Terms[:5])
            for j in range(len(Classifier) - 1):
                # Table.add_column(Classifier[j + 1], value2[j, 3:5])
                Table.add_column(Classifier[j + 1], value2[len(Algorithm) + j - 1, :5])
            print('-------------------------------------------------- Dataset -', str(i + 1), 'Batch Size-', str(Batch[m]),
                  ' -  Algorithm Comparison',
                  '--------------------------------------------------')
            print(Table)


def plot_results_Epoch():
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all_Epoch.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 1, 2, 3, 4]
    Algorithm = ['DMO', 'LOA', 'NRO', 'DPOA', 'IDPOA']
    Classifier = ['CNN', 'Resnet', 'RAN', '3DAEAN', 'PROPOSED']

    for i in range(No_of_Dataset):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
            Epoch = [50, 100, 150, 200, 250]
            plt.plot(Epoch, Graph[:, 0], color='#505250', linewidth=5, marker='D', markerfacecolor='skyblue', markeredgecolor="orange", markersize=12,
                     label="DMO-3DAEANet")
            plt.plot(Epoch, Graph[:, 1], color='#da6cf0', linewidth=5, marker='D', markerfacecolor='red', markeredgecolor="orange", markersize=12,
                     label="LOA-3DAEANet")
            plt.plot(Epoch, Graph[:, 2], color='#ff8e38', linewidth=5, marker='D', markerfacecolor='green', markeredgecolor="orange", markersize=12,
                     label="NRO-3DAEANet")
            plt.plot(Epoch, Graph[:, 3], color='#38fff2', linewidth=5, marker='D', markerfacecolor='cyan', markeredgecolor="orange", markersize=12,
                     label="DPOA-3DAEANet")
            plt.plot(Epoch, Graph[:, 4], color='#ff38b9', linewidth=5, marker='D', markerfacecolor='black', markeredgecolor="orange", markersize=12,
                     label="It-PDPO-3DAEANet")
            plt.xticks(Epoch, ('50', '100', '150', '200', '250'))

            plt.xlabel('Epochs')
            plt.ylabel(Terms[Graph_Term[j]])
            # plt.tick_params(axis='x', labelrotation=25)
            # plt.ylim([60, 100])
            plt.legend(loc=4)
            path1 = "./Results/Dataset-%s-%s_Alg_line.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


            fig = plt.figure()
            # ax = plt.axes(projection="3d")
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='#ff9900', width=0.10, edgecolor='k', label="CNN")
            ax.bar(X + 0.10, Graph[:, 6], color='#663300', width=0.10, edgecolor='k', label="ResNet")
            ax.bar(X + 0.20, Graph[:, 7], color='#ff0066', width=0.10, edgecolor='k', label="Unet")
            ax.bar(X + 0.30, Graph[:, 8], color='#0000ff', width=0.10, edgecolor='k', label="EAN")
            ax.bar(X + 0.40, Graph[:, 9], color='#006666', width=0.10, edgecolor='k', label="It-PDPO-3DAEANet")
            plt.xticks(X + 0.10,
                       ('50', '100', '150', '200', '250'))
            plt.xlabel('Epochs')
            # ax.tick_params(axis='x', labelrotation=45)
            plt.ylabel(Terms[Graph_Term[j]])
            # plt.ylim([60, 100])
            plt.legend(loc=1)
            path1 = "./Results/Dataset-%s-%s_Met_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


if __name__ == '__main__':
    plotConvResults()
    Plot_ROC_Curve()
    plot_results_Batch()
    plot_results_Epoch()
