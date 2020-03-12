# Run this script while inside data folder
import os
import glob
import sh

# CUR_DIR = os.cwd()
unzip = sh.Command('unzip')
mv = sh.Command('mv')
rm = sh.Command('rm')

for zipfile in glob.glob('*.zip'):
	dir_name = zipfile.split('.')[0]
	os.mkdir(dir_name)
	mv(zipfile, dir_name)

	os.chdir(dir_name)
	unzip(zipfile)
	# rm(zipfile)
	os.chdir('..')

# rm('*.zip')


