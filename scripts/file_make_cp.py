import glob
import os

for i in glob.glob('*.gjf'):
    print i
    os.system('mkdir '+i.replace('.gjf', ''))
    os.system('cp '+i+' '+i.replace('.gjf', '')+'/')
