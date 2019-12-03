# Bashscript to evaluate script individal_nr_ocr.py. Writes text file that contains all files that do not 
# have a positive ocr result yet

# IFS (Internal Field Seperator) is used by the shell to determine how to do word splitting,
# i. e. how to recognize word boundaries.
# Inside dollared single quotes, some characters are evaluated specially i.e. \n is translated to newline.
# The following line assigns newline to the variable IFS thus new line is used for word splitting instead of 
# space, tab  and newline (IFS default). This is necessary because of whitespaces used in filenames.
 IFS=$'\n'

# 'find' searches recursively all folders and subfolders of folder frogpictures_umi for all files with 
# .spnr file extension. Iterate over these files and
for file in $(find /home/stine/frogpictures_umi/ -name *.spnr)
do
	# read the file content and assign to ocred variable
	ocred=$(cat $file)
	
	# find speciesnumber in filename and assign to filenr variable
	filenr=$(echo $file | grep -o "201[0-9]\{5\}")
	# assign collection year to variable year (following 4 characters from position 2)
	year=${filenr:0:4}
	# assign collection number to variable num (following 4 character from position 6)
	num=${filenr:4:4}
	# assing speciesnumber as depicted in pictures (UA.year-num) to correct
	correct="UA."$year"-"$num
	
	# if ocred is unknown go to next file
	if [ "$ocred" == "unknown" ]
	then
		# increment variable unknown
		((unknown++))
		# write file with no match in variable nomatchfile
		nomatchfile=$(sed 's/\(.*\)\.spnr/\1/' <<< $file)
		# append nomatchfile to file nomatchfile.txt
		echo $nomatchfile>>nomatchfile.txt
		continue
	fi
	
	# replace all non ascii characters with -
	# ocred=${ocred//[^[:ascii:]]/-}
	# replace all characters between pattern with -
	ocred=$(sed 's/.*\([4-5]\{1\}\).*\([0-9]\{4\}\).*/UA.201\1-\2/' <<< $ocred)
	# replace all O, Q, o, G, C, D with 0
	ocred=$(sed 's/[OQoGCD]/0/g' <<< $ocred)
	
	# if ocred equals correct
	if [ "$ocred" == "$correct" ]
	then
		# increment variable matched
		((matched++))
	
	# if no match is found
	else
		# print ocred and correct for visual comparison
		# echo $ocred', '$correct
		
		# write file with no match in variable notmatchfile
		nomatchfile=$(sed 's/\(.*\)\.spnr/\1/' <<< $file)
		# append nomatchfile to file nomatchfile.txt
		echo $nomatchfile>>nomatchfile.txt
		((notmatched++))
		
	fi
	
	
	
done

# print number of matches and number of unknown
echo 'Number of matches:' $matched
echo 'Number of unknown:' $unknown
echo 'Number of not matched:' $notmatched
	




