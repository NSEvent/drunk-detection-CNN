# Split the data in train, validation, and test
# Run this script while inside data folder
import os
import glob
import random
import math
import sh
import shutil

################################
# How data is split
TRAIN_WEIGHT = 0.65
VALIDATION_WEIGHT = 0.15
TEST_WEIGHT = 0.2

# Where original data is
HOME = os.getcwd()

# Where split data will be saved
DIR_NAME = 'split_data'
################################

# Gather and randomize subjects
subjects = []
for person_dir in glob.glob('*'):
    if os.path.isdir(person_dir):
        subjects.append(person_dir)
        
random.shuffle(subjects)

# Split subjects
sz = len(subjects)
train = subjects[0:math.floor(TRAIN_WEIGHT*sz)]
validation = subjects[math.floor(TRAIN_WEIGHT*sz):math.floor((TRAIN_WEIGHT+VALIDATION_WEIGHT)*sz)]
test = subjects[math.floor((TRAIN_WEIGHT+VALIDATION_WEIGHT)*sz):]

# Ensure there are no overlaps
x = train + validation + test
x = set(x)

assert(len(x) == len(subjects))

# Create train, validation, and test folders
os.chdir('..')
if os.path.exists(DIR_NAME):
    shutil.rmtree(DIR_NAME)
os.mkdir(DIR_NAME)
os.chdir(DIR_NAME)

# Use soft link to original data to save space
ln = sh.Command('ln')

# Populate train folder
os.mkdir('train')
os.chdir('train')
for subject in train:
    folder_path = f'{HOME}/{subject}'
    ln('-sfv', folder_path, subject)
os.chdir('..')

# Populate validation folder
os.mkdir('validation')
os.chdir('validation')
for subject in validation:
    folder_path = f'{HOME}/{subject}'
    ln('-sfv', folder_path, subject)
os.chdir('..')

# Populate test folder
os.mkdir('test')
os.chdir('test')
for subject in test:
    folder_path = f'{HOME}/{subject}'
    ln('-sfv', folder_path, subject)
os.chdir('..')




