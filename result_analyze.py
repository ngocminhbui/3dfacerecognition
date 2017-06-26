from exp_config import *
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import  os
import pandas as pd


from sklearn.metrics import confusion_matrix

DATA_DIR = '/media/ngocminh/DATA/rgbd-dataset-processed-4dpng/'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    np.set_printoptions(precision=3)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def save_predictions_to_csv(predicts, targets, eval_list, dictionary):
    p = [(eval_list[i],dictionary[int(targets[i])],dictionary[int(predicts[i])]) for i in range(len(predicts))]
    df = pd.DataFrame.from_records(p)

    print df
    df.to_csv('~/Desktop/evalresult.csv')


def plot_false_predictions(predicts, targets, eval_list, dictionary):
    p = [[eval_list[i], i] for i in range(len(predicts)) if predicts[i] != targets[i]]
    for pi in p:
        file = pi[0]
        index = pi[1]

        trueclass = dictionary[int(targets[index])]
        predict = dictionary[int(predicts[index])]

        path = DATA_DIR + file + '_crop.png'

        command = "cp " + path + " ~/Desktop/false/" + trueclass + '_' + predict + '_' + str(index) + ".png"

        print command
        os.system(command)



f = open('./lists/dictionary.lst', 'r')
dictionary = f.read().split()
class_names = dictionary[:10]
score = np.loadtxt('./log/eval_mini_1-1/score_14001.txt')
predicts,targets= score[:,10],score[:,11]


# Compute confusion matrix
cnf_matrix = confusion_matrix(targets,predicts)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


f = open('./lists/eval_mini_1.lst', 'r')
eval_list = f.read().split()
plot_false_predictions(predicts,targets,eval_list, dictionary)


save_predictions_to_csv(predicts,targets,eval_list, dictionary)