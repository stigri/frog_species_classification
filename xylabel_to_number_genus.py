## same code as xylabel_to_number.py but in this case only images belonging to a specific target genus are rescaled and data
## structures X, y, labeltonumber are created. Only the distorted rescale method is used.
##

import pyexiv2
import sys
import os
import magic
import re
import cv2
import numpy as np

## lists are yspecies and ygenus from plot_inventory_exif.py
genera = ['Leptophryne', 'Microhyla', 'Fejervarya', 'Polypedates', 'Limnonectes', 'Chalcorana', 'Rhacophorus',
          'Nyctixalus', 'Amnirana', 'Phrynoidis', 'Odorrana', 'Ingerophrynus', 'Sumaterana', 'Pelophryne', 'Pulchrana',
          'Occidozyga', 'Hylarana', 'Leptobrachium', 'Megophrys', 'Huia', 'Philautus', 'Ansonia']
species = ['borbonica', 'heymonsi', 'achatina', 'limnocharis', 'leucomystax', 'otilophus', 'macrotis', 'paramacrodon',
           'sisikdagu', 'blythii', 'hikidai', 'kuhlii', 'macrodon', 'chalconota', 'rufipes', 'prominanus',
           'cyanopunctatus', 'catamitus', 'margaritifer', 'poecilonotus', 'pictum', 'nicobariensis', 'asper',
           'juxtasper', 'hosii', 'parvus', 'dabulescens', 'crassiovis', 'montana', 'signata', 'debussyi', 'rawa',
           'glandulosa', 'picturata', 'sumatrana', 'erythraea', 'hasseltii', 'nasuta', 'masonii']

## Function to rescale the images from rectangular to squared images with given edge lenght.
## Longer edge of original edge is resized to given edge length without distortion.
## Black borders are added to shorter edge to get squared format
# def pad(image, length):
#     ## get length and width of given image
#     origlength, origwidth = image.shape[:2]
#     ## if image is portrait format set length to given length without distortion
#     if origlength > origwidth:
#         ratio = length / float(origlength)
#         resizedwidth = int(origwidth * ratio)
#         dimensions = (resizedwidth, length)
#         resized = cv2.resize(image, dimensions, interpolation=cv2.INTER_CUBIC)
#         ## calculate size of borders to get squared image and add black border
#         left = int((length - resizedwidth) / 2)
#         right = left
#         top = 0
#         bottom = 0
#         resized = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
#
#
#     else:
#         ## if image is landscape format set width to given length without distortion
#         ratio = length / float(origwidth)
#         resizedlength = int(origlength * ratio)
#         dimensions = (length, resizedlength)
#         resized = cv2.resize(image, dimensions, interpolation = cv2.INTER_CUBIC)
#         ## calculate size of borders to get squared image and add black borders
#         left = 0
#         right = 0
#         top = int((length - resizedlength) / 2)
#         bottom = top
#         resized = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
#     ## return squared image with borders
#     return resized

def rescale(X, y, labeltonumber, absolutimpath, label, length):
    ## Use two approaches to get squared images:
    ## function pad: set long side of image to square length and add black or random noise to borders
    ## function resize: resize the image without caring about distortions
    ## size must be 299,299,3 for Xception
    origimage = cv2.imread(absolutimpath)

    smallimage = cv2.resize(origimage, (length, length), interpolation=cv2.INTER_CUBIC)

    ## append all resized images in array and label numbers in other array at same index
    X.append(smallimage)
    y.append(labeltonumber.index(label))
    return X, y


## rescale images to squared images and create numpy arrays from images and labels
def prepare_images(directory, length, target):
    labeltonumber = list()
    X = list()
    y = list()
    ## walk directories to find all images
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith((".JPG", ".jpg")):
                ## get absolutepath because of git annex
                absolutimpath = os.path.realpath(os.path.join(root, name))
                if not os.path.exists(absolutimpath):
                    # print('path %s is a broken symlink' % name)
                    pass
                elif not magic.from_file(absolutimpath).startswith("JPEG image data"):
                    pass
                else:
                    ## read metadata from exif tags of images
                    metadata = pyexiv2.ImageMetadata(absolutimpath)
                    metadata.read()
                    tag = metadata['Exif.Photo.UserComment']
                    genus_species = tag.value
                    genus_species = genus_species.strip()
                    match = re.search(r'(\w+)\s(\w+)', genus_species)
                    genus = match.group(1)
                    genus = genus.strip()
                    ## if given mode is genus create dictionary with all genera as keys and numbers 1-n as value
                    # if mode == 'genus':
                    #     label = genus
                    #     if label not in genera:
                    #         pass
                    #     else:
                    #         if label not in labeltonumber:
                    #             labeltonumber.append(label)
                    #         X, y = rescale(X, y, labeltonumber, absolutimpath, label)

                        ## make sure genus sp. are not included in species dict
                    ## if given mode is species create dictionary with all species as keys and numbers 1-n as value
                    if genus == target:
                        label = genus_species
                        ## do not use images were species is unknown
                        if match.group(2) in species:
                            if label not in labeltonumber:
                                labeltonumber.append(label)
                            X, y = rescale(X, y, labeltonumber, absolutimpath, label, length)
                    # else:
                    #     ## print error if mode is different from genus or species
                    #     sys.stderr.write(
                    #         'Usage: inventory_exif.py <path to directory containing .jpg files>, <pad> or <distort>,'
                    #         ' <length of resized image in pixel>, <width of resized image in pixel>, <genus> or <species>\n')



    ## create numpy array from each image in array of images
    X = np.array(X)
    y = np.array(y)
    labeltonumber = np.array(labeltonumber)
    # plt.imshow(X[1])
    # plt.show()
    print(len(X))
    print(len(y))
    ## return array of numpy array images and corresponding array of label numbers
    return X, y, labeltonumber


## safe X, y, labeltonumber on hard disc (numpy savez)
## shuffle data randomly before splitting
## use stratified sampling to devide training and test sets (training and test)
## sklearn.model_selection train_test_split (see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
## add another dict to show how many images there are of each label an return min value to
## k for k-fold cross validation (look into stratified sampling first)
## use stratified k-fold to generate folds
## in each iteration balance the training set not the validation fold
## function random minority oversampling: copy minority images to balance the data and return balanced dataset
## in each iteration use data augmentation in the training set not the validation fold (keras)
## normalize data (keras)
## use validation fold for validation
## use test set for final validation


if len(sys.argv) != 4:
    sys.stderr.write(
        'Usage: xylabel_to_number_genus.py <path to directory containing .jpg files>,'
        ' <edge length of resized image in pixel>, <target genus>\n')
    sys.exit(1)
else:
    directory = sys.argv[1]
    length = int(sys.argv[2])
    target = sys.argv[3]

X, y, labeltonumber = prepare_images(directory, length, target)
print(X)
print(y)
print(labeltonumber)
## compress files and save as .npz file
np.savez_compressed('data_{}_distort'.format(target), X = X, y = y, labeltonumber = labeltonumber)
