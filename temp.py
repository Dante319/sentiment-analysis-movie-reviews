import numpy as np
import pandas as pd
import nltk
import csv

from summa.summarizer import summarize

#file = open("C:/Users/Dante/PycharmProjects/Compiler/Datasets/aclImdb/test/neg/0_2.txt", "r", encoding='UTF-8')
import glob
neg = "C:/Users/Dante/PycharmProjects/Compiler/Datasets/aclImdb/test/neg/"
pos = "C:/Users/Dante/PycharmProjects/Compiler/Datasets/aclImdb/test/pos/"
outfile = open("C:/Users/Dante/PycharmProjects/Compiler/Datasets/aclImdb/test/test_res.csv", "w+", newline='')
count=0
for files in glob.glob(neg +"*.txt"):
    count = count+1
    infile = open(files, errors='ignore')
    text = infile.read()
    res = summarize(text, ratio=0.2)
    temp = ""
    for line in res:
        temp += line.rstrip('\n')
    CSVWriter = csv.writer(outfile)
    CSVWriter.writerow(['0', str(temp)])
    #CSVWriter.writerow(temp)
    #outfile.write(temp)
    #print(str(count) + ": " + res + " " + files)
    #print('--------')

for files in glob.glob(pos +"*.txt"):
    count = count+1
    infile = open(files, errors='ignore')
    text = infile.read()
    res = summarize(text, ratio=0.2)
    temp = ""
    for line in res:
        temp += line.rstrip('\n')
    CSVWriter = csv.writer(outfile)
    CSVWriter.writerow(['1', str(temp)])

print ("done")

