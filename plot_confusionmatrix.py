## code original from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
## script to plot heatmap of normalized cofusion matrix

import itertools
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys



def load_data(filename):
    print('[INFO] loading data...')
    npzfile = np.load(filename)
    labeltonumber = npzfile['labeltonumber']
    # print(X)
    # print(y)
    # print(labeltonumber)
    return labeltonumber


def plot_confusion_matrix(cm, classes, ax1, cax, ax2):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # if normalize:
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cmn = np.multiply(cmn, 100)
    cmn = np.around(cmn)
    cmn = cmn.astype('int')
    cm = cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    # ax2 = fig.add_subplot(133)
    # cax = fig.add_subplot(131)
    im1 = ax1.imshow(cmn, aspect='auto', interpolation='nearest', cmap=plt.cm.jet)
    ax2.imshow(cm, aspect='equal', interpolation='nearest', cmap=plt.cm.jet)
    # else:
    #     print('Confusion matrix, without normalization')
    #     plt.imshow(cm, aspect='auto', interpolation='nearest', cmap=plt.cm.YlOrRd)


    print(cmn)
    maxcm = int(max(cm))
    # cbar = ax1.figure.colorbar(im1, ax=cax)
    cbar = plt.colorbar(im1, cax=cax)
    cax2 = cax.twinx()
    ax2ticks = np.arange(0, maxcm + 10, 25)
    ax1ticks = np.arange(0, 101, 10)
    cbar.set_ticks(ax1ticks)
    cbar.set_label("%")
    cbar.ax.yaxis.set_label_position('left')
    # cbar.set_yticklabels(ax1ticks)
    cax2.set_yticks(ax2ticks)
    cax2.set_yticklabels(ax2ticks)
    cax2.set_ylabel("Number images")
    # cbar.tick_params(axis = 'y', direction = 'out', right = True, labelright = True)
    # cax2.tick_params(axis = 'y', direction = 'out', left = True, labelleft = True)
    cax2.set_aspect




    # ax1.set_title('Normalized Confusion matrix')
    ax1.set_xticks(np.arange(len(classes)))
    ax1.set_xticklabels(classes, {'horizontalalignment': 'right'})
    ax1.tick_params(axis='x',
                    direction='out',
                    labelrotation=30)
    ax1.set_yticks(np.arange(len(classes)))
    ax1.set_yticklabels(classes)

    font = {'weight': 'bold'}
    threshlow = cmn.max() / 4.
    threshhigh = cmn.max() / 5 * 4

    # if normalize:
    # fmt = '.2f'
    for i, j in itertools.product(range(cmn.shape[0]), range(cmn.shape[1])):
        ax1.text(j, i, format(cmn[i, j]),
                    horizontalalignment="center",
                    color="black" if cmn[i, j] > threshlow and cmn[i, j] < threshhigh else "white",
                    fontdict=font if cmn[i, j] > 0 and cmn[i, j] < 100 else None)
    # else:
    #     fmt = 'd'
    #     sum = cm.sum(axis=1)[:, np.newaxis]
    #     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #         plt.text(j, i, format(cm[i, j], fmt),
    #                  horizontalalignment="center",
    #                  color="white" if cm[i, j] > threshhigh else "black",
    #                  fontdict=font if cm[i, j] >= 0 and cm[i, j] < sum[i] else None)

    ax1.set_ylabel('True label')
    ax1.set_xlabel('Predicted label')
    print('Number of images')
    # ax2.set_title('Number of images')
    ax2.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        left=False,
        right=False,
        labelleft=False
    )
    ax2.set_ylabel('Number images')


    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax2.text(j, i, format(cm[i, j]),
                    horizontalalignment="center",
                    color="black" if cm[i, j] <= 200 and cm[i, j] > 62 else "white")

plt.rcParams.update({'font.size': 10})

## parameter determine the training version
if len(sys.argv) != 4:
    sys.stderr.write(
        'Usage: plot_confusionmatrix.py [species|genus], [pad|distort], <version>\n')
    sys.exit(1)
else:
    mode = sys.argv[1]
    resize = sys.argv[2]
    version = sys.argv[3]

## frogsumimodels/Xcetpion_[genus|species]_[pad|distort]_$version
path = 'frogsumimodels/Xception_{}_{}_{}/test_{}_{}_{}.pkl'.format(mode, resize, version, mode, resize, version)

with open(path, 'rb') as f:
    accuracy, classreport, cnf_matrix, math_corrcoef, y_prob, y_pred = pickle.load(f)

print('loss: {}, accuracy: {}'.format(accuracy[0], accuracy[1]))
print('MCC: {}'.format(math_corrcoef))
filename = 'npz/data_{}_{}.npz'.format(mode, resize)
labeltonumber = load_data(filename)
print(classreport)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=labeltonumber,
#                       title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
fig, (ax2, ax1, cax) = plt.subplots(ncols=3, gridspec_kw={'width_ratios': [1, 4, 0.1]})
fig.set_figheight(17)
fig.set_figwidth(30)
plot_confusion_matrix(cnf_matrix, labeltonumber, ax1, cax, ax2)
plt.show()
