## Script to plot the inventory of what genera and species are shown on the images as shown in Figure 5.3.1

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import pickle

## needs dictinventory.pkl created by inventory_exif.py
## two plots with the same x-axis next to each other (four plots in total)
## list: genus (y-axis)
## list: species (y-axis)
## list: number of individuals > 3 (x-axis)
## list: number of pictures > 3 (x-axis)
## value: mean number of pictures genus (horizontal line)
## value: mean number of pictures species (excluding sp.) (horizontal line)
## code runs in pythontex

## read the data from pkl file
with open('/home/stine/repositories/MSCCode/dictinventory.pkl', 'rb') as di:
    dictgenusspecies, dictspeciesspnr, dictspnrpath = pickle.load(di)

## initialize figure1 and two with given sizes
## initialize four axes objects, with right y-axis invisible
fig1 = plt.figure(figsize=(10, 4))

ax1 = SubplotHost(fig1, 121)
ax1.axis["right"].set_visible(False)

ax2 = SubplotHost(fig1, 122)
ax2.axis["right"].set_visible(False)

fig2 = plt.figure(figsize=(10, 10))
ax3 = SubplotHost(fig2, 121)
ax3.axis["right"].set_visible(False)

ax4 = SubplotHost(fig2, 122)
ax4.axis["right"].set_visible(False)

## add axes objects to figures
fig1.add_subplot(ax1)
fig1.add_subplot(ax2)

fig2.add_subplot(ax3)
fig2.add_subplot(ax4)

## define colors that are to be used in plots
colors = ['maroon', 'red', 'orange', 'yellow', 'cyan', 'lime', 'teal', 'darkkhaki', 'plum', 'magenta', 'purple', 'navy',
          'sienna', 'pink', 'darkorchid', 'deeppink', 'gray', 'gold', 'olive', 'tomato', 'green', 'lavender', 'black']

## initialize variables needed and explained later
idx = 0
y = 0
x = 0
ygenus1 = list()
ygenus2 = list()
xnrindividualgen = list()
xnrimagesgen = list()
ypos = list()
yspecies = list()
xnrindividualspec = list()
xnrimagesspec = list()
coloring = list()

## iterate over all genera in dictionary
for genus in dictgenusspecies:
    ## iterate over all species of the given genus in dictionary
    for species in dictgenusspecies[genus]:
        ## count number of individuals for species
        nrindividualspec = len(dictspeciesspnr[species])
        ## undifined species (genus sp.) are labeled as genus genus
        ## remove all undefined species
        if species not in dictgenusspecies.keys():
            ## remove all species with less then 3 individuals
            if nrindividualspec > 3:
                included = 1
                nrimagesspec = 0
                ## append species to species list that is to be plotted
                yspecies.append(species)
                ## define position in barplot, needed to get empty space between genera
                y += 1
                ypos.append(y)
                ## append number of individuals to list that is to be plotted
                xnrindividualspec.append(nrindividualspec)
                ## count number of images for species
                for spnr in dictspeciesspnr[species]:
                    nrimagesspec += len(dictspnrpath[spnr])
                ## append number of images to list that is to be plotted
                xnrimagesspec.append(nrimagesspec)
                ## append color to list of colors to get the same color for each species of the same genus
                coloring.append(colors[idx])

    if included == 1:
        y += 1
        idx += 1
        ## append genus to list of genus1, the list of genera which have defined species included to keep order of
        ## genera in same order of genera in species plot
        if genus not in ygenus1:
            ygenus1.append(genus)
    else:
        nrindividualgen = len(dictspeciesspnr[species])
        ## remove all genera that are not included in species (because all individuals are genus sp.) with less
        ## then 3 individuals
        if nrindividualgen > 3:
            if genus not in ygenus2:
                ## add all genera with more than three individuals to second genera list
                ygenus2.append(genus)

    included = 0

    ## plot horizontal barplot for species with given color and label (second barplot without species label)
    ax3.barh(ypos, xnrindividualspec, color=coloring, tick_label=yspecies)
    ax4.barh(ypos, xnrimagesspec, color=coloring, tick_label='')

## calculate the min and max number of individuals and images, draw vertical line with value on upper x-axis
## for species barplots
minnrindividualspec = min(xnrindividualspec)
maxnrindividualspec = max(xnrindividualspec)
minnrimagesspec = min(xnrimagesspec)
maxnrimagesspec = max(xnrimagesspec)
ax3.axvline(minnrindividualspec, color='black', linestyle='--')
ax3.text(minnrindividualspec - 2, 61.7, minnrindividualspec)
ax3.axvline(maxnrindividualspec, color='black', linestyle='--')
ax3.text(maxnrindividualspec - 5, 61.7, maxnrindividualspec)
ax4.axvline(minnrimagesspec, color='black', linestyle='--')
ax4.text(minnrimagesspec - 20, 61.7, minnrimagesspec)
ax4.axvline(maxnrimagesspec, color='black', linestyle='--')
ax4.text(maxnrimagesspec - 30, 61.7, maxnrimagesspec)

## append the two lists of genus to get one list to be plotted in correct order
ygenus = ygenus1 + ygenus2
## iterate over genera in generated list of genera
for genus in ygenus:
    nrindividualgen = 0
    nrimagesgen = 0
    ## iterate over species in dictionaty belonging to genera in list
    for species in dictgenusspecies[genus]:
        ## count number of indiviuals for each genus
        nrindividualgen += len(dictspeciesspnr[species])
        ## iterate over spnr in dictionary of selected species
        for spnr in dictspeciesspnr[species]:
            ## count number of images per genera
            nrimagesgen += len(dictspnrpath[spnr])

    ## add number of individuals and of images per genera to respective list to be plotted
    xnrindividualgen.append(nrindividualgen)
    xnrimagesgen.append(nrimagesgen)

## plot horizontal barplot for given genera with given color and label (second plot without label on y-axis)
ax1.barh(ygenus, xnrindividualgen, color=colors, tick_label=ygenus)
ax2.barh(ygenus, xnrimagesgen, color=colors, tick_label='')


## calculate mean number of images and individuals and plot vertical line in genera barplots
minnrindividualgen = min(xnrindividualgen)
maxnrindividualgen = max(xnrindividualgen)
minnrimagesgen = min(xnrimagesgen)
maxnrimagesgen = max(xnrimagesgen)
ax1.axvline(minnrindividualgen, color='black', linestyle='--')
ax1.text(minnrindividualgen - 2, 22.8, minnrindividualgen)
ax1.axvline(maxnrindividualgen, color='black', linestyle='--')
ax1.text(maxnrindividualgen - 8, 22.8, maxnrindividualgen)
ax2.axvline(minnrimagesgen, color='black', linestyle='--')
ax2.text(minnrimagesgen - 20, 22.8, minnrimagesgen)
ax2.axvline(maxnrimagesgen, color='black', linestyle='--')
ax2.text(maxnrimagesgen - 35, 22.8, maxnrimagesgen)

# fig1.suptitle('Original data composition of collected genera', fontsize=14)
# fig2.suptitle('Original data composition of collected species', fontsize=14)
## label x-axis of all plots
ax1.xaxis.set_label_text("Number individuals")
ax2.xaxis.set_label_text("Number images")
ax3.xaxis.set_label_text("Number individuals")
ax4.xaxis.set_label_text("Number images")
## activate grid in all plots
ax1.grid(1)
ax2.grid(1)
ax3.grid(1)
ax4.grid(1)
## show all plots
## in lyx document barplots are saved here and included in latex document
plt.show()
