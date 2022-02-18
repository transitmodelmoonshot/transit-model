import os
from inspect import getsourcefile
from os.path import abspath
import openpyxl #opens excel files
def dump_to_excel(values):
    directory = abspath(getsourcefile(lambda:0))
    #check if system uses forward or backslashes for writing directories
    if(directory.rfind("/") != -1):
        newDirectory = directory[:(directory.rfind("/")+1)]
    else:
        newDirectory = directory[:(directory.rfind("\\")+1)]
    os.chdir(newDirectory)
    
    path = "predictions.xlsx"
    wb = openpyxl.load_workbook(path)
    sh = wb["data"]
    i = 1
    for value in values:
        sh["A{}".format(i)]=value
        i=i+1
    wb.save(path)
    return
dump_to_excel([1,2,3])
