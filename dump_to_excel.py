import os
from inspect import getsourcefile
from os.path import abspath
import openpyxl #opens excel files
def dump_to_excel(values,col="A"):
    directory = abspath(getsourcefile(lambda:0))
    #check if system uses forward or backslashes for writing directories
    if(directory.rfind("/") != -1):
        newDirectory = directory[:(directory.rfind("/")+1)]
    else:
        newDirectory = directory[:(directory.rfind("\\")+1)]
    os.chdir(newDirectory)

    path = "predictions.xlsx"
    wb = openpyxl.load_workbook(path)
    sh = wb.active
    i = 1
    for value in values:
        sh["{}{}".format(col,i)]=value
        i=i+1
    wb.save(path)
    return
