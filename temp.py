from os import listdir
from os.path import isfile, join
import pathlib
import csv
import os

def truncate(path):
    dirs = [f for f in listdir(path)]
    for d in dirs:
        print("Starting for dir {}".format(d))
        files = [f for f in listdir(join(path, d)) if isfile(join(join(path, d), f))]
        for f in files:
            data = []
            count = 0
            with open(join(join(path, d), f)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if count < 20:
                        data.append(row)
                    if len(row) > 0 and row[0] == "Run complete":
                        count += 1
                
            if count == 1:
                print("removing {}".format(join(join(path, d), f)))
                os.remove(join(join(path, d), f))
            elif count == 21:
                print("truncating {}".format(join(join(path, d), f)))
                with open(join(join(path, d), f), mode='w') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) 
                    csv_writer.writerows(data)
            elif count != 20:
                print("weird {} {}".format(join(join(path, d), f), count))

def main():
    path = "benchmarks/argon"
    truncate(path + "/forward")
    truncate(path + "/inverse")

if __name__ == "__main__":
    main()
