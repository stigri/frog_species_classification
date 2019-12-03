## Code to write spnr, genus and species in exif data / xmp tags.

import pyexiv2
import sys
import os
import magic
import re

##
## Exif information is written in:
## Exif.Image.ImageDescription = spnr
## Exif.Photo.UserComment = Genus species
## xmp tags are written in:
## Xmp.mscbioinf.individualID = spnr
## Xmp.mscbioinf.genus = genus
## Xmp.mscbioinf.species = species
## TODO: Xmp data has not been written to all files yet since information for first run was already in exif data.
## TODO: Use as guideline when additional information from excel files need to be stored in pictures.
## To be able to write tags because of git annex one needs to unlock all files (git annex unlock *)
## Once you are done they need to be added (git annex add *), locked (git annex lock *),
## synchronized (git annex sync) and copied (git annex copy -t hetzner.de .)

## function to initially label images in exif tags as described above
def labelexif(directory):
        for root, dirs, files in os.walk(directory):
            for name in files:
                if name.endswith((".JPG", ".jpg")):
                    ## get absolutepath because of git annex
                    absolutimpath = os.path.realpath(os.path.join(root, name))
                    absolutimpathspnr = os.path.realpath(os.path.join(root, name) + '.spnr')
                    ## complicated way to deal with git annex, last elif would be enough as criteria to run code
                    if not os.path.exists(absolutimpathspnr):
                        # print('path %s is a broken symlink' % name)
                        pass
                    elif not os.path.exists(absolutimpath):
                        pass
                    elif not magic.from_file(absolutimpath).startswith("JPEG image data"):
                        pass
                    else:
                        metadata = pyexiv2.ImageMetadata(absolutimpath)
                        metadata.read()

                        with open(absolutimpathspnr) as spnr:
                            lines = [line.rstrip('\n') for line in spnr]
                            metadata['Exif.Image.ImageDescription'] = lines[0]
                            species = '{} {}'.format(lines[1], lines[2])
                            metadata['Exif.Photo.UserComment'] = species
                            print(metadata['Exif.Image.ImageDescription'])
                            print(metadata['Exif.Photo.UserComment'])
                            metadata.write()

## function to correct exif tags. There was a literal error in one species name that needed to be corrected
def correctexif(directory):
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith((".JPG", ".jpg")):
                ## get absolutepath because of git annex
                absolutimpath = os.path.realpath(os.path.join(root, name))
                absolutimpathspnr = os.path.realpath(os.path.join(root, name) + '.spnr')
                ## complicated way to deal with git annex, last elif would be enough as criteria to run code
                if not os.path.exists(absolutimpathspnr):
                    # print('path %s is a broken symlink' % name)
                    pass
                elif not os.path.exists(absolutimpath):
                    pass
                elif not magic.from_file(absolutimpath).startswith("JPEG image data"):
                    pass
                else:
                    metadata = pyexiv2.ImageMetadata(absolutimpath)
                    metadata.read()

                    with open(absolutimpathspnr) as spnr:
                        lines = [line.rstrip('\n') for line in spnr]
                        metadata['Exif.Image.ImageDescription'] = lines[0]
                        species_name = 'borbonica '
                        species = '{} {}'.format(lines[1], species_name)
                        metadata['Exif.Photo.UserComment'] = species
                        print(metadata['Exif.Image.ImageDescription'])
                        print(metadata['Exif.Photo.UserComment'])
                        metadata.write()
                        file = open(root + "/" + name + '.spnr', 'w')
                        file.write(lines[0])
                        file.write('\n')
                        file.write(lines[1])
                        file.write('\n')
                        file.write(species_name)
                        file.close()

## function to create customized xmp tags and write information in tags.
## Did not run the code on my pictures yet will do when adding additional information from data sheets
def write_xmp(directory):
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith((".JPG", ".jpg")):
                ## get absolutepath because of git annex
                absolutimpath = os.path.realpath(os.path.join(root, name))
                ## complicated way to deal with git annex, last elif would be enough as criteria to run code
                if not os.path.exists(absolutimpath):
                    pass
                elif not magic.from_file(absolutimpath).startswith("JPEG image data"):
                    pass
                else:
                    ## read metadata
                    metadata = pyexiv2.ImageMetadata(absolutimpath)
                    metadata.read()
                    ## read exif tags that I wrote ealier
                    tag = metadata['Exif.Photo.UserComment']
                    genus_species = tag.value
                    match = re.search(r'(\w+)\s(\w+)', genus_species)
                    genus = match.group(1)
                    species = match.group(2)
                    tag2 = metadata['Exif.Image.ImageDescription']
                    idvnr = tag2.value
                    ## register namespace to be able to invent customized xmp tags like 'species', 'genus', etc
                    ## namespace is registered now and can be used as shown below
                    #pyexiv2.xmp.register_namespace('http://griep.at/mscbioinf/', 'mscbioinf')
                    ## write data in customized xmp tags
                    metadata['Xmp.mscbioinf.individualID'] = idvnr
                    metadata['Xmp.mscbioinf.genus'] = genus
                    metadata['Xmp.mscbioinf.species'] = species
                    metadata.write()





## run code by entering path to image directory
if len(sys.argv) != 2:
    sys.stderr.write(
        'Usage: write_exifdata.py <path to image directory>\n')
    sys.exit(1)
else:
    directory = sys.argv[1]



labelexif(directory)
# correctexif(directory)
# write_xmp(directory)