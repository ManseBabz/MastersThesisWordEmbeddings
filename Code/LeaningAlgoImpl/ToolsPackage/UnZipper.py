import zipfile, codecs
import numpy.random as ran

class ZippedSentences(object):
    def __init__(self, zippedname, number_to_extract, random=False):
        self.zippedname = zippedname
        self.number_to_extract = number_to_extract
        self.random = random


    def printinfo(self):
        i = 0
        with zipfile.ZipFile(self.zippedname) as myzip:
            for file in myzip.filelist:
                if i <= self.number_to_extract:
                    i += 1
                    with myzip.open(file) as myfile:
                        #myfile = myfile.decode("utf-8")
                        for line in myfile:
                            line = codecs.decode(line, "utf-8")
                            print(line)

    def __iter__(self):
        i = 0
        with zipfile.ZipFile(self.zippedname) as myzip:
            if(self.random==True):
                list_of_files_to_use = ran.choice(myzip.filelist, size=self.number_to_extract, replace=True)
            else:
                list_of_files_to_use = myzip.filelist
            for file in list_of_files_to_use:
                if i <= self.number_to_extract:
                    i += 1
                    with myzip.open(file) as myfile:
                        for line in myfile:
                            line = codecs.decode(line, "utf-8")
                            yield line.split()