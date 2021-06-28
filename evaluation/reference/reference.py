from os import listdir
from os.path import isfile, join
import re
import csv
import matplotlib.pyplot as plt

options = {"tc1": 2, "tc2": 3, "tc3": 3}

prefix = "../../benchmarks_krypton/reference"
P1 = 2; P2 = 2
cuda_aware = False
prec = "double"

titles = ["Bandwidth MPI Communication", "Bandwidth Slab Transpose", "Bandwidth Pencil Transpose"]
legend = [["Peer2Peer", "All2All"], ["Synchronize", "Streams", "Custom MPI_Type"], ["Synchronize", "Streams", "Custom MPI_Type"]]

for tc in range(1, 4):
    labels = []
    for opt in range(options["tc" + str(tc)]):
        opt_dir = prefix + "/testcase" + str(tc) + "/opt" + str(opt)
        files = [f for f in listdir(opt_dir) if isfile(join(opt_dir, f)) and re.match("test_\d*_{}_{}_{}_{}".format(P1, P2, str(cuda_aware).lower(), prec), f)]
        
        x_vals = []
        y_vals = []

        for f in files:
            with open(join(opt_dir, f)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_counter = 0
                for row in csv_reader:
                    if tc == 1:
                        if line_counter == 1:
                            x_vals.append(1/(P1*P2) * sum([float(row[i+1]) for i in range(P1*P2)]))
                        elif line_counter == 2:
                            y_vals.append(1/(P1*P2) * sum([float(row[i+1]) for i in range(P1*P2)]))
                    else:
                        if line_counter == 1:
                            x_vals.append(1/(P1*P2) * sum([float(row[i+1]) for i in range(P1*P2)]))
                        elif line_counter == 2:
                            x_vals[-1] = (x_vals[-1] + 1/(P1*P2) * sum([float(row[i+1]) for i in range(P1*P2)])) / 2
                        elif line_counter == 3:
                            y_vals.append(1/(P1*P2) * sum([float(row[i+1]) for i in range(P1*P2)]))
                        elif line_counter == 4:
                            y_vals[-1] = (y_vals[-1] + 1/(P1*P2) * sum([float(row[i+1]) for i in range(P1*P2)])) / 2
                    line_counter += 1

        y_vals = [x for _, x in sorted(zip(x_vals, y_vals), key=lambda pair: pair[0])]
        x_vals = sorted(x_vals)
        label, = plt.plot(x_vals, y_vals, 'D-', linewidth=3, zorder=3)
        labels.append(label)
    
    plt.title(titles[tc-1], fontsize=20)
    plt.xlabel("Size [MB]", fontsize=20)
    plt.ylabel("Bandwidth [MB/s]", fontsize=20)
    plt.legend(labels, legend[tc-1], prop={"size":16})
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(zorder=0, color="grey")
    plt.show()
    plt.close()




