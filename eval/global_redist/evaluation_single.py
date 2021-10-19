# Copyright (C) 2021 Simon Egger
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from os import listdir
from os.path import isfile, join
import pathlib
import re
import csv
import numpy as np
from scipy.stats import t
import sys


prefix = "benchmarks/bwunicluster/gpu4"
if len(sys.argv) > 1:
    prefix = "benchmarks" + str(sys.argv[1])
prec = "double"

run_iterations = 20

def getSizes(subdir):
    files = [f for f in listdir(join(prefix, subdir)) if isfile(join(join(prefix, subdir), f)) and re.match("test_0_0_0_\d*_\d*_\d*_0_1", f)]
    sizes = []
    for f in files:
        sizes.append(f.split("_")[4] + "_" + f.split("_")[5] + "_" + f.split("_")[6])
    sizes.sort(key=lambda e: int(e.split("_")[0]))
    sizes.sort(key=lambda e: int(e.split("_")[1]))
    sizes.sort(key=lambda e: int(e.split("_")[2]))
    return sizes

def reduceRun(size, subdir):
    file = join(join(prefix, subdir), "test_0_0_0_{}_0_1.csv".format(size))
    
    data = -1
    sd = -1

    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        iterations = 0
        for row in csv_reader:
            if len(row) > 0:
                if row[0] == "Run complete":
                    iterations += 1
                    if data == -1:
                        data = 0
                    data += float(row[1])
        if data != -1:
            data /= iterations

    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        iterations = 0
        for row in csv_reader:
            if len(row) > 0:
                if row[0] == "Run complete":
                    iterations += 1
                    if sd == -1:
                        sd = 0
                    sd += (float(row[1])-data)**2
        if sd != -1:
            sd = np.sqrt(sd/(iterations-1))
        
    return data, sd 

def reduceTestcase(forward, subdir):
    data = []
    sd = []

    sizes = getSizes(subdir)

    for size in sizes:
        run_data, run_sd = reduceRun(size, subdir)
        if run_data != -1 and run_sd != -1:
            data.append(run_data)
            sd.append(run_sd)

    pathlib.Path("eval/{}".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
    with open('eval/{}/runs.csv'.format(join(prefix, subdir)), mode='w') as out_file:
        writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(sizes)
        writer.writerow(data)
    with open('eval/{}/sd.csv'.format(join(prefix, subdir)), mode='w') as out_file:
        writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(sizes)
        writer.writerow(sd)
                
def main():
    for forward in range(0, 2):
        subdir = "{}/single".format("forward" if forward==1 else "inverse")
        pathlib.Path("eval/{}".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
        
        reduceTestcase(forward, subdir)

if __name__ == "__main__":
    main()
