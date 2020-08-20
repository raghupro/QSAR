import os
fileDir = os.path.dirname(os.path.realpath('__file__'))
dirPath = os.path.join(fileDir, 'logs')
print('Filedir:', fileDir)
print('DirPath:', dirPath)
print('pwd:', os.getcwd())
