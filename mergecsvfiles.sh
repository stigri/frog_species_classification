# Bashscript that merges different .csv files that contain collection data to one .csv file that contains all collection data belonging to the images used in the thesis.

# IFS (Internal Field Seperator) is used by the shell to determine how to do word splitting,
# i. e. how to recognize word boundaries.
# Inside dollared single quotes, some characters are evaluated specially i.e. \n is translated to newline.
# The following line assigns newline to the variable IFS thus new line is used for word splitting instead # of space, tab  and newline (IFS default). This is necessary because of whitespaces used in filenames.
IFS=$'\n'

# 'find' iterates recursively over all folders and subfolders of folder frogpictures_umi/csv, whish was the folder that contained all collection data .csv files. 
for file in $(find /home/stine/frogpictures_umi/csv/ -name *.csv)
do
	# the second (individual number) and third column (species) of each .csv file is read and the 		# content is written into one .csv (mergedcsvfild.csv) file that contains all individual numbers 		# and corresponding species name	
	csvtool col 2,3 $file| grep -o "UA\.201[4-5]\{1\}\.[0-9]\{4\}.*" >>/home/stine/frogpictures_umi/csv/mergedcsvfile.csv
done
