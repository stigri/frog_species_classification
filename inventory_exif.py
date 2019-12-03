## Script to get inventory of which genera and species are shown on the images and how many individuals there are per
## genus and species.

import pyexiv2
import sys
import os
import magic
import re
from collections import defaultdict
import pickle


## generates dictinventory.pkl which is used in plot_inventory_exif.py
## TODO: Use as guideline to write dictidvnrsvl, dictidvnrweight, dictidvnrgps, etc.
## TODO: (Do not read metadata from exif then but from xmp data)



def inventory_exif(directory):
    dictgenusspecies = defaultdict(list)
    dictspeciesidvnr = defaultdict(list)
    dictidvnrpath = defaultdict(list)
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
                    ## read metadata from exif tags
                    metadata = pyexiv2.ImageMetadata(absolutimpath)
                    metadata.read()
                    tag = metadata['Exif.Photo.UserComment']
                    genus_species = tag.value
                    match = re.search(r'(\w+)\s(\w+)', genus_species)
                    genus = match.group(1)
                    ## make sure that genus sp ist saved for each genus by calling genus sp = genus genus
                    if match.group(2) == 'sp':
                        species = genus
                    else:
                        species = match.group(2)
                    # if species == 'barbonica':
                    #     print(absolutimpath)
                    tag2 = metadata['Exif.Image.ImageDescription']
                    idvnr = tag2.value

                    ## create dictionaries with information:
                    ## dictgenus = key: genus, value: species
                    ## dictspecies = key: species, value: idvnr
                    ## dictidvnr = key: idvnr, value: absolutimpath
                    if genus not in dictgenusspecies.keys():
                        ## if key genus not in dictgenus add information to all dicts
                        dictgenusspecies[genus].append(species)
                        dictspeciesidvnr[species].append(idvnr)
                        dictidvnrpath[idvnr].append(absolutimpath)
                    elif species not in dictspeciesidvnr.keys():
                        ## if key species not in dictspecies add information to all dicts
                        dictgenusspecies[genus].append(species)
                        dictspeciesidvnr[species].append(idvnr)
                        dictidvnrpath[idvnr].append(absolutimpath)
                    elif idvnr not in dictidvnrpath.keys():
                        ## if idvnr not in dictspnr add idvnr to dictspecies and idvnr and path to dictspnr
                        dictspeciesidvnr[species].append(idvnr)
                        dictidvnrpath[idvnr].append(absolutimpath)
                    else:
                        ## if idvnr already in dictspnr, add path to dictspnr
                        dictidvnrpath[idvnr].append(absolutimpath)

    with open('dictinventory.pkl', 'wb') as di:
        pickle.dump([dictgenusspecies, dictspeciesidvnr, dictidvnrpath], di)

    ## show what genus and species, how many individuals and number of pictures per species I got
    nrpictavspec = 0
    nrpictavgen = 0
    for genus in dictgenusspecies:
        nrindgen = 0
        for species in dictgenusspecies[genus]:
            nrindgen += len(dictspeciesidvnr[species])
            nrindspec = len(dictspeciesidvnr[species])
            nrpict = 0
            for idvnr in dictspeciesidvnr[species]:
                nrpict += len(dictidvnrpath[idvnr])
            if nrindspec > 3:
                print('{}, {}, {}, {}'.format(genus, species, nrindspec, nrpict))
                if species != 'sp':
                    nrpictavspec += nrpict
            nrpictavgen += nrpict
    print(nrpictavspec / len(dictspeciesidvnr))
    print(nrpictavgen / len(dictgenusspecies))
    print(len(dictspeciesidvnr))
    print(len(dictgenusspecies))

















if len(sys.argv) != 2:
    sys.stderr.write(
        'Usage: inventory_exif.py <path to directory containing .jpg files>\n')
    sys.exit(1)
else:
    directory = sys.argv[1]

inventory_exif(directory)