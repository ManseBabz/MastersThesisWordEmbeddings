import zipfile, codecs

class ZippedSentences(object):
    def __init__(self, zippedname, number_to_extract):
        self.zippedname = zippedname
        self.number_to_extract = number_to_extract


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
            for file in myzip.filelist:
                if i <= self.number_to_extract:
                    i += 1
                    with myzip.open(file) as myfile:
                        for line in myfile:
                            line = codecs.decode(line, "utf-8")
                            yield line.split()