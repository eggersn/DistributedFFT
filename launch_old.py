import json
import os, sys
from os import listdir
from os.path import isfile, join


def menu_main():
    valid = False
    os.system("clear")
    while not valid:
        print("Select a Category:\n" +
        "-" * 35 + "\n" + 
        "[0] Run Specified Job (job.json)\n" +
        "[1] Pencil Decomposition \n" + 
        "[2] Slab Decomposition \n\n" +
        "Selection: ")
        try: 
            opt = int(input())
            if opt >= 0 and opt <= 2:
                return opt    
        except ValueError: 
            os.system("clear")
            print("Please enter an integer!\n")
        except KeyboardInterrupt:
            print("\nexiting...")
            sys.exit()

    return 0

def select_job():
    valid = False
    os.system("clear")
    jobs = [f for f in listdir("jobs") if isfile(join("jobs", f))]
    while not valid:
        print("Select a Job:\n" + "-" * 35)
        for i in range(len(jobs)):
            print("[{}] {}".format(i, jobs[i]))
        print("Selection: ")
        try: 
            opt = int(input())
            if opt >= 0 and opt < len(jobs):
                return jobs[opt]
        except ValueError: 
            os.system("clear")
            print("Please enter an integer!\n")
        except KeyboardInterrupt:
            print("\nexiting...")
            sys.exit()

    return 0

if len(sys.argv) == 1:
    opt = menu_main()
    if opt == 0:
        job = select_job()
        filename = "../jobs/" + job
else:
    opt = int(sys.argv[1])
    filename = "../jobs/" + sys.argv[2] + ".json"

hostname = os.uname()[1]
if hostname[0:6] == "pcsgs0":
    os.chdir("build_pcsgs")
elif hostname == "krypton":
    os.chdir("build_krypton")
print()

if opt == 0:
    with open(filename) as f:
        data = json.load(f)

    for s in data["size"]:
        print("Starting computation for size {}".format(s))
        for test in data["tests"]:
            ranks = 0
            if test["name"][0:6] == "Pencil" or test["name"] == "Reference":
                print("-> Executing test {} (case {}, opt {})\n   (size: {}, P1: {}, P2: {}, cuda_aware: {}, precision: {})".format(test["name"], test["testcase"], test["option"], s, test["P1"], test["P2"], str(test["cuda_aware"]).lower(), test["precision"]))
                ranks = test["P1"] * test["P2"]
            else:
                print("-> Executing test {} \n   (size: {}, P: {}, cuda_aware: {}, precision: {})".format(test["name"], s, test["P"], str(test["cuda_aware"]).lower(), test["precision"]))
                ranks = test["P"]
            # One additional process for coordination
            if test["testcase"] == 1 and test["name"] != "Reference":
                ranks += 1
            command = "mpiexec -n " + str(ranks) + " " + data["additional-flags"] + " " + "tests_exec " + str(test["name"]) + " " + str(test["testcase"]) + " " 
            command += str(test["option"]) + " " + str(data["warmup-rounds"] + test["repetitions"]) + " " + str(s) + " " + str(s) + " " + str(s) + " " + str(test["cuda_aware"]).lower() + " " + test["precision"]
                
            benchmarkfile = test["name"].lower()
            if test["name"][0:6] == "Pencil":
                command += " " + str(test["P1"]) + " " + str(test["P2"])
                benchmarkfile = "pencil"
            elif test["name"] == "Reference":
                command += " " + str(test["P1"]) + " " + str(test["P2"])
                benchmarkfile = "reference"

            if test["option"] != 0:
                benchmarkfile += "_opt" + str(test["option"])

            os.system(command)

            if test["name"] == "Reference":
                if not os.path.isdir("../benchmarks/reference"):
                    os.system("mkdir ../benchmarks/reference")
                if not os.path.isdir("../benchmarks/reference" + "/testcase" + str(test["testcase"])):
                    os.system("mkdir ../benchmarks/reference"+ "/testcase" + str(test["testcase"]))
                if not os.path.isdir("../benchmarks/reference" + "/testcase" + str(test["testcase"]) + "/opt" + str(test["option"])):
                    os.system("mkdir ../benchmarks/reference" + "/testcase" + str(test["testcase"]) + "/opt" + str(test["option"]))
            else:
                if not os.path.isdir("../benchmarks/" + benchmarkfile):
                    os.system("mkdir ../benchmarks/" + benchmarkfile)

            with open("../benchmarks/" + benchmarkfile + ".csv", 'r') as f_in:
                f_data = f_in.read().splitlines(True)
                index = 0
                if test["name"] != "Reference":
                    index = [i for i, n in enumerate(f_data) if n == "\n"][data["warmup-rounds"]-1]
            if test["name"] == "Reference":
                with open("../benchmarks/reference" + "/testcase" + str(test["testcase"]) + "/opt" + str(test["option"]) +  "/test_" + str(s) + "_" + str(test["P1"]) + "_" + str(test["P2"]) + "_" + str(test["cuda_aware"]).lower() + "_" + test["precision"]  + ".csv", 'w') as f_out:
                    f_out.writelines(f_data[0])
                    f_out.writelines(f_data[index+1:])
                os.system("rm ../benchmarks/" + benchmarkfile + ".csv")
            elif test["name"][0:6] == "Pencil":
                with open("../benchmarks/" + benchmarkfile + "/test_" + str(s) + "_" + str(test["P1"]) + "_" + str(test["P2"]) + "_" + str(test["cuda_aware"]).lower() + "_" + test["precision"]  + ".csv", 'w') as f_out:
                    f_out.writelines(f_data[0])
                    f_out.writelines(f_data[index+1:])
                os.system("rm ../benchmarks/" + benchmarkfile + ".csv")
            else:
                with open("../benchmarks/" + benchmarkfile + "/test_" + str(s) + "_" + str(test["P"]) + "_" + str(test["cuda_aware"]).lower() + "_" + test["precision"] + ".csv", 'w') as f_out:
                    f_out.writelines(f_data[0])
                    f_out.writelines(f_data[index+1:])
                os.system("rm ../benchmarks/" + benchmarkfile + ".csv")
            
        print()
elif opt == 1:
    print("todo")
else:
    print("todo")


