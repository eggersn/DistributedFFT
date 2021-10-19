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
import os
import pathlib
import re
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


prefix = "benchmarks/argon"

files = [f for f in listdir(prefix) if isfile(join(prefix,f)) and (f[-3:]=="out" or f[-3:]=="txt") ]
index_map = ["Peer2Peer-Sync", "Peer2Peer-Streams", "Peer2Peer-MPI_Type", "All2All-Sync", "All2All-MPI_Type"]
buffer = {}
sizes = []

files.sort()

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

                if len(row.split(": ")) == 3:
                    buffer[P][name][index_map.index("{}-{}".format(comm, snd))][size] = [float(row.split(": ")[1].split("\\n")[0]), float(row.split(": ")[2].split("\\n")[0])]
                elif len(row.split(": ")) == 2:
                    buffer[P][name][index_map.index("{}-{}".format(comm, snd))][size] = [0, 0]
                    buffer[P][name][index_map.index("{}-{}".format(comm, snd))][size][0] = float(row.split(": ")[1].split("\\n")[0])
                    row = next(out_file)
                    buffer[P][name][index_map.index("{}-{}".format(comm, snd))][size][1] = float(row.split(": ")[1].split("\\n")[0])

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


