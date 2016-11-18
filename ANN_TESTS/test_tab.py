# This script will detect all tab characters.

import os # os-dependent functionality
import re # regular expressions

# Directories to check
os.chdir("../ANN")
path = os.getcwd()

# Grep for the pattern in given files.
fileList = []

def grep(pattern, fileList):
    forPass = 0 
    for file in fileList:
        tabsCount = 0
        f = open(file, 'r')
        content = f.readlines()     
        for line in content:
            if re.search(pattern,line):
                tabsCount += 1
                forPass   += 1 
        if tabsCount != 0:        
            print("FAIL: Detected %d tab characters in file %s." %(tabsCount, file))
    if forPass == 0:
        print("PASS: All spaces.")  
    f.close()
  
# List of file types to check.
files = os.listdir(path)
for f in files:
    file = os.path.join(path, f)
    if ( file.endswith(".h")) or (file.endswith(".cpp")) or (file.endswith(".txt") ):
        fileList += [file]

grep("\t", fileList)
           