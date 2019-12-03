## Skript to check plausibility of ocr spnr and to correct wrongly ocr'd spnr
## Does not find all errors, probably because not all possible cases are covered.


import sys
import os
import cv2
import scipy.misc
from shutil import copyfile
import re

## function to write new .spnr file if ocr was wrong
def writenewspnrfile(pathlist, idx, image):
    ## enter correct year and input
    year = input('Enter year: 4/5')
    number = input('Enter number')
    ## create new idvnr and show image again for confirmation
    idvnr = 'UA.201' + year + number
    print(idvnr)
    cv2.imshow('case 1', image)
    cv2.waitKey()
    ## check if correct
    correct = input('correct? y/n')
    ## if not correct run same function again
    if correct == 'n':
        writenewspnrfile(pathlist, idx, image)
    ## else open merged csv, get corresponding species and write both in new .spnr file
    else:
        stream = open(mergedcsv, 'r')
        for line in stream:
            csv_match = re.search(r'(UA\.201[4,5]{1}\.\d{4}),(.*)', line)
            csv_idvnr = csv_match.group(1)
            csv_species = csv_match.group(2)
            if idvnr == csv_idvnr:
                print('{} name: {} species: {}, speciesNr: {}'.format(csv_species, csv_idvnr))
                ## writes ocr result in .spnr file
                file = open(pathlist[idx], 'w')
                file.write(csv_idvnr)
                file.write('\n')
                file.write(csv_species)
                file.close()
    return idvnr


def plausibilitycheck(path):
    ## min number of images that have the same spnr (dorsal, lateral, ventral)
    mincount = 3
    ## max number of images that have the same spnr
    maxcount = 5
    count = 0
    ## list of all spnr
    spnrlist = list()
    ## list of all path to spnr files
    pathlist = list()
    ## walk directories to find all .spnr files
    for root, dirs, files in os.walk(path):
        for name in sorted(files):
            if name.endswith(".spnr"):
                ## get absolutepath because of git annex
                absolutimpath = os.path.realpath(os.path.join(root, name))
                ##print(absolutimpath)

                ## get backup copy of each .spnr file
                copyfile(absolutimpath, absolutimpath+'.bak')
                ## open .spnr files and copy spnr and path to corresponding lists
                with open(absolutimpath) as idvnr:
                    idvnr = idvnr.readline()
                    idvnr = idvnr.rstrip()
                    spnrlist.append(idvnr)
                    pathlist.append(absolutimpath)

    ## iterate over lists
    for idx, idvnr in enumerate(spnrlist):
        pathspnr = pathlist[idx]
        ## get path to image file of current .spnr file
        pathimage = pathspnr[:-5]
        ## exception for first element in list
        if idvnr == 'unknown':
            print(pathspnr)
        elif idx == 0:
            ## print first spnr and open corresponding picture to check if correct
            print('first spnr: {}'.format(idvnr))
            image = scipy.misc.imread(pathimage, 'L')
            image = scipy.misc.imresize(image, 10, 'cubic')
            cv2.imshow('case 1', image)
            cv2.waitKey()
            ## ask user to enter y or n if it is correct or not
            correct = input('correct? y/n')
            ## if not write new .spnr file
            if correct == 'n':
                spnrlist[idx] = writenewspnrfile(pathlist, idx, image)
                count = 1
            ## else count first element with this number
            else:
                count = 1
        ## for all elements but the first in the list
        else:
            ## get lastidvnr (spnr before current) and nextidvnr (spnr next in the list, behind current)
            lastidvnr = spnrlist[idx - 1]
            nextidvnr = spnrlist[idx + 1]
            ## if spnr equals lastidvnr, increment count for number euqal spnr in a row
            if idvnr == lastidvnr:
                count = count + 1
                ## if count is higher than max count, show image and check if it is correct, if not write new .spnr
                if count > maxcount:
                    print(pathimage)
                    print(idvnr)
                    image = scipy.misc.imread(pathimage, 'L')
                    image = scipy.misc.imresize(image, 10, 'cubic')
                    cv2.imshow('case 1', image)
                    cv2.waitKey()

                    correct = input('count > maxcount. correct? y/n')
                    if correct == 'y':
                        count = count + 1
                        continue
                    else:
                        spnrlist[idx] = writenewspnrfile(pathlist, idx, image)
                        count = 1
            elif count >= mincount:
                if idvnr == nextidvnr:
                    count = 1
                else:
                    ## if different but lastidvnr and nextidvnr are the same
                    if lastidvnr == nextidvnr:
                        print('Case 1: surroundingidvnr {}, currentidvnr {}'.format(lastidvnr, idvnr))

                        ## show image to see if spnr matches last and next
                        image = scipy.misc.imread(pathimage, 'L')
                        image = scipy.misc.imresize(image, 10, 'cubic')
                        cv2.imshow('case 1', image)
                        cv2.waitKey()

                        idvnr = input('Case 1: same as surroundingidvnr? y/n/c')
                        ## if it matches, copy last .spnr file and save copy as current, increment count
                        if idvnr == 'y':
                            copyfile(pathlist[idx - 1], pathlist[idx])
                            count = count + 1
                        ## some pictures were in wrong order but correctly ocr, to skip these press c for continue
                        elif idvnr == 'c':
                            continue
                        ## if picture not the same as surrounding images
                        else:
                            ## show surrounding images to see if they are correctly ocr
                            pathlastimage = pathlist[idx-1][:-5]
                            pathnextimage = pathlist[idx+1][:-5]
                            image = scipy.misc.imread(pathlastimage, 'L')
                            image = scipy.misc.imresize(image, 10, 'cubic')
                            cv2.imshow('case 1.2', image)
                            cv2.waitKey()
                            image = scipy.misc.imread(pathnextimage, 'L')
                            image = scipy.misc.imresize(image, 10, 'cubic')
                            cv2.imshow('case 1.2', image)
                            cv2.waitKey()
                            ## if surrounding images are wrong and current image is right, rewrite .spnr for surrounding images
                            surroundingwrong = input('are surrounding images wrong? y/n')
                            if surroundingwrong == 'y':
                                copyfile(pathlist[idx], pathlist[idx-1])
                                copyfile(pathlist[idx], pathlist[idx+1])
                                count = 2
                            ## if surrounding images are correct and current aswell show path to check
                            else:
                                ## else write correct spnr in .spnr file
                                ##writenewspnrfile(pathlist, idx, image)
                                ##count = 1
                                print(pathlist[idx])

            ## if new spnr and count is lower than mincount
            elif count < mincount:
                ## show lastidvnr, currentspnr and nextidvnr
                print('Case 2: lastidvnr {}, currentidvnr {}, nextidvnr {}'.format(lastidvnr, idvnr, nextidvnr))

                ## open image to see correct current spnr
                image = scipy.misc.imread(pathimage, 'L')
                image = scipy.misc.imresize(image, 10, 'cubic')
                cv2.imshow('case 2', image)
                cv2.waitKey()

                ## ask if correct spnr equals last or next spnr, if so make copy of respective .spnr file and save as current .spnr file and adjust count
                idvnr = input('Case 2: lastidvnr: l, nextidvnr: n, other: o')
                # if spnr == 'l':
                #     copyfile(pathlist[idx-1], pathlist[idx])
                #     count = count + 1
                if idvnr == 'n':
                    print(pathlist[idx-1])
                    print(pathlist[idx])
                    count = 1
                ## if not write new .spnr file with correct number and species and adjust count
                elif idvnr == 'l':
                    copyfile(pathlist[idx - 1], pathlist[idx])
                    count = count + 1
                else:
                    spnrlist[idx] = writenewspnrfile(pathlist, idx, image)
                    count = 1




        print(count)




## run code by entering the directory where .spnr files are and where merged csv file is
if len(sys.argv) != 3:
    sys.stderr.write(
        'Usage: idvnr_plausibility.py <path to directory containing .spnr files, path to merged csv>\n')
    sys.exit(1)
else:
    directory = sys.argv[1]
    mergedcsv = sys.argv[2]


plausibilitycheck(directory)
