## original code to cite: https://github.com/raghakot/keras-vis/blob/master/examples/resnet/attention.ipynb
## script to calculate gradCAM and to show heatmaps and misclassified images of one system
## reads img_attr, calculates and visualizes gradcam for attribution. Saves cam heatmap images in .npz file to overlay with original images
## code only runs if model is trained on cpu or single gpu. If trained parallely on multiole gpus it does not run.

import tensorflow as tf
from keras.applications import Xception
from keras import activations
from vis.utils import utils
import sys
import numpy as np
from vis.visualization import visualize_cam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import heapq


## load X, y, labeltonumber
def load_all_data(filename):
    print('[INFO] loading data...')
    npzfile = np.load(filename)
    X = npzfile['X']
    y = npzfile['y']
    labeltonumber = npzfile['labeltonumber']
    print(len(X))
    print(len(y))
    # print(labeltonumber)
    return X, y, labeltonumber


## function to load images img_attr containing one image of each class of test set. GradCAM was finally calculated for
## all images so this function was not used.
def load_img_attr_data(images):
    print('[INFO] loading images...')
    npzfile = np.load(images)
    img_attr = npzfile['img_attr']
    return img_attr


def load_img_heatmaps(heatmaps):
    print('[INFO] loading heatmaps...')
    npzfile = np.load(heatmaps)
    img_heatmaps = npzfile['img_heatmap']
    # print(img_heatmaps.shape)
    return img_heatmaps


## reads parameter that define which images and weights will be used for attribution and if you want to calculate gradCAM
## and save respective heatmaps or if you want to show heatmaps or misclassified images.
if len(sys.argv) != 6:
    sys.stderr.write(
        'Usage: plot_heatmap.py [species|genus], [pad|distort], [save|show], <version>, <weightfile>\n')
    sys.exit(1)
else:
    mode = sys.argv[1]
    resize = sys.argv[2]
    modus = sys.argv[3]
    version = sys.argv[4]
    weightfile = sys.argv[5]

path_weights = 'frogsumimodels/Xception_{}_{}_{}/'.format(mode, resize, version)
# images = 'npz/img_attr_{}_{}.npz'.format(mode, resize)
images = 'npz/data_{}_{}.npz'.format(mode, resize)
heatmaps = 'npz/img_heatmap_{}_{}_{}.npz'.format(mode, resize, version, mode, resize, version)
# img = load_img_attr_data(images)
X, y, labeltonumber = load_all_data(images)

path_results = 'frogsumimodels/Xception_{}_{}_{}/test_{}_{}_{}.pkl'.format(mode, resize, version, mode, resize, version)

with open(path_results, 'rb') as f:
    accuracy, classreport, cnf_matrix, math_corrcoef, y_prob, y_pred = pickle.load(f)

## normalizes images used for attribution
X = X.astype('float32') / 255
mean = X.mean(axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
# print(y_test)
X_test_mean = X_test - mean

if modus == 'save':

    ## for tests use data_{}_{}.test.npz
    ## /home/stine/repositories
    # filename = 'npz/img_attr_{}_{}.npz'.format(mode, resize)
    # img_attr = load_img_attr_data(filename)

    ## creates model used for training and loads weights
    print('[INFO] create model and load weights ...')
    weights = path_weights + weightfile
    # with tf.device('/cpu:0'):
    model = Xception(include_top=True, weights=weights, classes=len(labeltonumber))

    # model = multi_gpu_model(model, gpus=2)
    # model.load_weights(weights)

    ## changes last layer activation from softmax to linear to improve attribution results
    print('[INFO] change activations of last layer to linear ...')
    layer_idx = utils.find_layer_idx(model, 'predictions')
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    print(model.summary())
    print(model.input)

    print('[INFO] calculate cam ...')
    ## defines last convolutional layer before dense layers
    penultimate_layer = utils.find_layer_idx(model, 'block14_sepconv2')

    img_heatmap = []
    ## iterates over all images in array
    for idx, img in enumerate(X_test):
        print('image shape: {}'.format(img.shape))
        # plt.imshow(img)
        # plt.show()
        print('norm image shape: {}'.format(X_test_mean[idx].shape))
        ## generates a gradient based class activation map (grad-CAM) that maximizes the outputs of filter_indices in layer_idx
        ## returns the heatmap image indicating the input regions whose change would most contribute towards maximizing the output of filter_indices
        grads = visualize_cam(model, layer_idx, filter_indices=y_test[idx], seed_input=X_test_mean[idx],
                              backprop_modifier='relu')
        print('grads shape: {}'.format(grads.shape))
        img_heatmap.append(grads)

    ## saves heatmap array as .npz file
    np.savez_compressed('img_heatmap_{}_{}_{}'.format(mode, resize, version), img_heatmap=img_heatmap)

elif modus == 'show':
    ## code to show all images that were misclassified (comment if you want to see heatmaps instead)
    # for idx, img in enumerate(X_test):
    #     if y_pred[idx] != y_test[idx]:
    #         label = labeltonumber[y_test[idx]]
    #         pred_label = labeltonumber[y_pred[idx]]
    #         prob = heapq.nlargest(3, zip(y_prob[idx], labeltonumber))
    #         first = prob[0]
    #         second = prob[1]
    #         third = prob[2]
    #         fig, ax = plt.subplots(1, 1)
    #         ax.imshow(img)
    #         fig.suptitle('{}\n{}: {}\n{}: {}\n{}: {}'.format(label, first[1], first[0], second[1], second[0], third[1],
    #                                                          third[0]))
    #         plt.show()

    ## code to show heatmaps (comment if you want to see misclassified images instead)
    img_heatmaps = load_img_heatmaps(heatmaps)
    print(len(img_heatmaps))
    for idx, img in enumerate(X_test):
        label = labeltonumber[y_test[idx]]
        # pred_label = labeltonumber[y_pred[idx]]
        prob = heapq.nlargest(3, zip(y_prob[idx], labeltonumber))
        first = prob[0]
        second = prob[1]
        third = prob[2]
        heatmap = img_heatmaps[idx]
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(img)
        # ax[1].imshow(overlay(heatmap, img))
        if label == 'Ingerophrynus':
            fig = plt.figure()
            im1 = plt.imshow(img)
            im2 = plt.imshow(heatmap, alpha=0.5)
            # fig.suptitle('{}\n{}: {}\n{}: {}\n{}: {}'.format(label, first[1], first[0] , second[1], second[0], third[1], third[0]))
            print('{}\n{}: {}\n{}: {}\n{}: {}'.format(label, first[1], first[0], second[1], second[0], third[1],
                                                      third[0]))

            plt.show()
