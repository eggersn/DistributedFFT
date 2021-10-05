from os import listdir
from os.path import isfile, join
import os
import pathlib
import re
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


prefix = "benchmarks/bwunicluster/gpu8/large"

files = [f for f in listdir(prefix) if isfile(join(prefix,f)) and f[-3:]=="out"]
index_map = ["Peer2Peer-Sync", "Peer2Peer-Streams", "Peer2Peer-MPI_Type", "All2All-Sync", "All2All-MPI_Type"]
buffer = {}
sizes = []

for f in files:
    with open(join(prefix,f)) as out_file:
        lines = ["", ""]
        for row in out_file:
            if "Result" in row and "mpiexec -n " in lines[1]:
                P = int(lines[1].split("mpiexec -n ")[1].split(" ")[0])
                if P not in buffer:
                    buffer[P] = {}
                name = ""; suffix = ""
                if "slab" in lines[1]:
                    name = "Slab"
                    if "Z_Then_YX" in lines[1]:
                        name += "_1D-2D"
                    else:
                        name += "_2D-1D"
                else:
                    name = "Pencil"
                    suffix = "_{}_{}".format(lines[1].split("-p1 ")[1].split(" ")[0], lines[1].split("-p2 ")[1].split(" ")[0])

                if "--opt 1" in lines[1]:
                    name += "_Realigned"
                else:
                    name += "_Default"
                name += suffix

                if name not in buffer[P]:
                    buffer[P][name] = [{} for i in range(5)]

                comm = ""; snd = ""
                if "All2All" in lines[1]:
                    comm = "All2All"
                else:
                    comm = "Peer2Peer"
                if "Streams" in lines[1]:
                    snd = "Streams"
                elif "MPI_Type" in lines[1]:
                    snd = "MPI_Type"
                else:
                    snd = "Sync"

                size = lines[1].split("-nx ")[1].split(" ")[0] + "_" + lines[1].split("-ny ")[1].split(" ")[0] + "_" + lines[1].split("-nz ")[1].split(" ")[0]
                if size not in sizes:
                    sizes.append(size)

                buffer[P][name][index_map.index("{}-{}".format(comm, snd))][size] = [float(row.split(": ")[1].split("\\n")[0]), float(row.split(": ")[2].split("\\n")[0])]

            lines[1] = lines[0]
            lines[0] = row 

sizes.sort(key=lambda e: int(e.split("_")[0]))
sizes.sort(key=lambda e: int(e.split("_")[1]))
sizes.sort(key=lambda e: int(e.split("_")[2]))

for P in buffer:
    with open('eval/{}/numerical_{}.csv'.format(prefix, P), mode='w') as out_file:
        writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Average"])
        for s in sizes:
            writer.writerow([s])
            for key in buffer[P]:
                writer.writerow([key] + [c[s][0] for c in buffer[P][key] if s in c])
            writer.writerow([])
        writer.writerow([])
        writer.writerow(["Maximum"])
        for s in sizes:
            writer.writerow([s])
            for key in buffer[P]:
                writer.writerow([key] + [c[s][1] for c in buffer[P][key] if s in c])
            writer.writerow([])


