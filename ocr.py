## code to run ocr on images to recognize individual numbers and write results into '.spnr' files

import numpy as np
import cv2
import os
import scipy.misc
import pytesseract
from PIL import Image
import re
import sys


## function to adjust gamma value of the image
def adjust_gamma(image, gamma=1.0):
    ## build a lookup table mapping the pixel values [0, 255] to
    ## their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


## function to recognize individual number in the image and write it to '.spnr' file
def individual_nr_ocr(inputimage):
    # scaling down picuture size
    # image = cv2.resize(image, None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
    # using  scipy.misc because of bug in cv2.resize (version 3.0.0) function
    # scaling down to 30% of original picture size using cubic interpolation
    image = scipy.misc.imresize(inputimage, 30, 'cubic')

    # adjust gamma
    gamma = 2.5
    image = adjust_gamma(image, gamma)

    #### comment this part if you do NOT want to use individual number plate extraction ####################################
    ## guideline: http://nbviewer.jupyter.org/gist/kislayabhi/89b985e5b78a6f56029a (12.06.2017)

    ## contrast limited adaptive histogram equalization (clahe) to enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    equalized = clahe.apply(image)

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', equalized)
    # cv2.waitKey()

    ## blur the image to reduce noise
    blur = cv2.GaussianBlur(equalized, (1, 1), 0)

    # cv2.namedWindow('Gaussian blur', cv2.WINDOW_NORMAL)
    # cv2.imshow('Gaussian blur', blur)
    # cv2.waitKey()

    ## use sobel gradient to find vertival edges within the image with kernel size 3
    sobel = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)

    # cv2.namedWindow('Sobel gradient', cv2.WINDOW_NORMAL)
    # cv2.imshow('Sobel gradient', sobel)
    # cv2.waitKey()

    ## use Otsu's thresholding to divide image into fore- and background
    ret, threshold = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #   cv2.namedWindow('Otsus thresholding', cv2.WINDOW_NORMAL)
    #   cv2.imshow('Otsus thresholding', threshold)
    #   cv2.waitKey()

    ##   morphological closing with cross shaped structuring element to shmooth lines
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 5))
    closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, se)

    #   cv2.namedWindow('Morphological closing', cv2.WINDOW_NORMAL)
    #   cv2.imshow('Morphological closing', closing)
    #   cv2.waitKey()

    ##   find contours of possible individual number plates
    _, contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    ##   validate a contour by estimating roughly area and aspect ratio of the actual individual number plate
    def validate(cnt):
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        output = False
        width = rect[1][0]
        height = rect[1][1]
        if ((width != 0) & (height != 0)):
            if ((height / width > 4) | (width / height > 4)):
                if ((height * width < 100000) & (height * width > 3000)):
                    output = True
        return output

    ##   crop and adjust rectangle of validated contours
    ##   https://stackoverflow.com/questions/37177811/crop-rectangle-returned-by-minarearect-opencv-python (12.06.2017)
    for cnt in contours:
        if validate(cnt):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            W = rect[1][0]
            H = rect[1][1]

            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)

            angle = rect[2]
            if angle < -45:
                angle += 90

            ## Center of rectangle in source image
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            ## Size of the upright rectangle bounding the rotated rectangle
            size = ((x2 - x1), (y2 - y1))
            M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

            ## cropped upright rectangle
            cropped = cv2.getRectSubPix(image, size, center)
            cropped = cv2.warpAffine(cropped, M, size)
            croppedW = H if H > W else W
            croppedH = H if H < W else W
            ## final cropped & rotated rectangle
            croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW + 100), int(croppedH + 100)),
                                               (size[0] / 2, size[1] / 2))
            ret, thresh = cv2.threshold(croppedRotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            ## morphological opening with rectangle shaped structuring element
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, se)

            # cv2.namedWindow('species number', cv2.WINDOW_NORMAL)
            # cv2.imshow('species number', thresh)
            # cv2.waitKey(5000)
            #### end comment this part if you do NOT want to use individual number plate extraction ################################

            ## find species numbers using tesseract ocr and filter for defined user pattern
            pattern = re.compile('([4,5]{1}).{0,2}(\d{4})')
            imgforocr = Image.fromarray(image)
            ## config= -psm 7 (single text line) --user patterns (UA.201\d-\d\d\d\d)
            ## config= -psm 6 (single block of text) or -psm 8 (single word) seem to work as well
            ## --user-patterns does not work as expected. Why?
            speciesnr_ocr = pytesseract.image_to_string(imgforocr,
                                                        config='-c tessedit_char_whitelist=.-UA0123456789 -psm 7')
            # print (speciesnr_ocr)
            match = re.search(pattern, speciesnr_ocr)

            ## reformat match
            if match:
                year = match.group(1)
                number = match.group(2)
                speciesNr = 'UA.201{}.{}'.format(year, number)
                # print('speciesNr: {}'.format(speciesNr))
                return speciesNr


# Function to walk through directory and find all .jpg files.
# Calls function individual_nr_ocr and writes ocr results in .spnr files
def walkDirectories(path):
    # initialize variable cntimages to count number of images and unknown to count number of files
    # without ocr result
    cntimages = 0
    unknown = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith((".JPG", ".jpg")):
                # abslotuimpath is the real path stored in symlink file (git annex)
                absolutimpath = os.path.realpath(os.path.join(root, name))
                # if symlink is broken, throw exeption, if not go on
                if not os.path.exists(absolutimpath):
                    # print('path %s is a broken symlink' % name)
                    pass
                elif os.path.isfile(root + "/" + name + '.spnr'):
                    # Skip successfully ocr'ed files from previous runs of the program.
                    pass
                else:
                    # reads image and calls function for individual number recognition
                    # using  scipy.misc because of bug in cv2.resize (version 3.0.0) function
                    image = scipy.misc.imread(absolutimpath, 'L')
                    ocr_individual_nr = individual_nr_ocr(image)
                    # count processedimges
                    cntimages = cntimages + 1
                    # count processed images where ocr did not work
                    if ocr_individual_nr is None:
                        unknown = unknown + 1
                    # open .csv to see if ocr result has match with real individual number and find species it belongs to
                    stream = open(mergedcsv, 'r')
                    for line in stream:
                        csv_match = re.search(r'(UA\.201[4,5]{1}\.\d{4}),(.*)', line)
                        csv_individual_nr = csv_match.group(1)
                        csv_species = csv_match.group(2)
                        if ocr_individual_nr == csv_individual_nr:
                            print('{} name: {} species: {}, speciesNr: {}'.format(cntimages, root + "/" + name,
                                                                                  csv_species, csv_individual_nr))
                            ## writes ocr result in .spnr file
                            file = open(root + "/" + name + '.spnr', 'w')
                            if csv_individual_nr is None:
                                file.write('unknown')
                            else:
                                file.write(csv_individual_nr)
                                file.write('\n')
                                file.write(csv_species)
                            file.close()

                            # print(os.path.join(root, name))
                            # print('{} img: {}, spnr: {}'.format(cntimages, name, ocr_individual_nr))
                            # print(speciesNr)
    print('cntimages: {}, unknown: {}'.format(cntimages, unknown))


## Function to read files with no match to call individual_nr_ocr function again with different gamma value or after
## extracting the individual number plate
def walkUnmatchedFiles(nomatchfile):
    ## initialize variable cntimages to count number of images and unknown to count number of files
    ## without ocr result
    cntimages = 0
    unknown = 0
    with open('nomatchfile.txt') as fp:
        for line in fp:
            ## remove \n from read line
            line = line.rstrip()
            ## reads image and calls function for individual number recognition
            ## using  scipy.misc because of bug in cv2.resize (version 3.0.0) function
            image = scipy.misc.imread(line, 'L')
            individual_nr = individual_nr_ocr(image)
            cntimages = cntimages + 1
            if individual_nr is None:
                unknown = unknown + 1
            ## writes ocr result in .spnr file
            file = open(line + '.spnr', 'w')
            if individual_nr is None:
                file.write('unknown')
            else:
                file.write(individual_nr)
            file.close()
            print(cntimages)
            print(line)
            print(individual_nr)
        print(unknown)


## run program by calling program name, path to image directory and path to mergedcsvfile.csv
if len(sys.argv) != 3:
    sys.stderr.write(
        'Usage: individual_nr_ocr.py <path to directory containing frogpictures> <path to mergedcsvfile.csv>\n')
    sys.exit(1)
else:
    directory = sys.argv[1]
    mergedcsv = sys.argv[2]
#   print(directory)

# path1 = '/home/stine/frogpictures_umi'

## nomachfile.txt is created with evaluate_ocr.sh
# path2 = '/home/stine/repositories/MSCCode/nomatchfile.txt'

# path3 = '/home/stine/frogpictures_umi.annex'

## mergedcsvfile.csv can be created by using mergescvfile.sh
# path4 = '/home/stine/frogpictures_umi/csv/mergedcsvfile.csv'


walkDirectories(directory)
# walkUnmatchedFiles(path2)
