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
    # jobs = [f for f in listdir("jobs") if isfile(join("jobs", f))]
    selected_jobs = []
    prefix = "jobs"
    while True:
        os.system("clear")
        valid = False
        jobs = [f for f in listdir(prefix)]
        while not valid:
            print("Select a Job:\n" + "-" * 35)
            print("[0] all")
            for i in range(len(jobs)):
                print("[{}] {}".format(i+1, jobs[i]))
            print("Selection: ")
            try: 
                opt = int(input())
                if opt == 0:
                    return [os.path.join(dp, f) for dp, dn, filenames in os.walk(prefix) for f in filenames if os.path.splitext(f)[1] == '.json']
                if opt > 0 and opt <= len(jobs):
                    if isfile(join(prefix, jobs[opt-1])):
                        return [prefix + "/" + jobs[opt-1]]
                    prefix += "/" + jobs[opt-1]
                    valid = True
            except ValueError: 
                os.system("clear")
                print("Please enter an integer!\n")
            except KeyboardInterrupt:
                print("\nexiting...")
                sys.exit()
    return 0

def keysToLower(test):
    for key in test:
        test[str(key).lower()] = test[key]

def convertKey(test, key1, key2):
    if (not key1 in test) and (key2 in test):
        test[key1] = test[key2]

def checkIfParamExists(test, key1, key2, error_msg=""):
    if (not key1 in test) and (not key2 in test):
        raise LookupError(error_msg)
    convertKey(test, key1, key2)

def run_test(test, size, global_test_settings, additional_flags, parse):
    if parse == True:
        test.update(global_test_settings)
        if size != 0:
            if "--input-dim-x" in test or "-nx" in test:
                print("Warning: Nx is overridden by size!")
            if "--input-dim-y" in test or "-ny" in test:
                print("Warning: Ny is overridden by size!")
            if "--input-dim-z" in test or "-nz" in test:
                print("Warning: Nz is overridden by size!")

            test["-nx"] = size
            test["-ny"] = size
            test["-nz"] = size

        convertKey(test, "--testcase", "-t")
        
        ranks = 0
        checkIfParamExists(test, "name", "name", "Invalid test: No name specified. Available names are [\"Reference\", \"Slab\", \"Pencil\"].")
        if size == 0:
            checkIfParamExists(test, "--input-dim-x", "-nx", "Invalid test: Nx not specified")
            checkIfParamExists(test, "--input-dim-y", "-ny", "Invalid test: Ny not specified")
            checkIfParamExists(test, "--input-dim-z", "-nz", "Invalid test: Nz not specified")
        if test["name"].lower() == "pencil":
            checkIfParamExists(test, "-p1", "--partition1", "Invalid test: P1 not specified")
            checkIfParamExists(test, "-p2", "--partition2", "Invalid test: P2 not specified")
            ranks = test["-p1"] * test["-p2"]
            if "--testcase" in test and test["--testcase"] == 1:
                ranks += 1
        elif test["name"].lower() == "slab":
            checkIfParamExists(test, "-p", "--partition", "Invalid test: P not specified")
            ranks = test["-p"]
            if "--testcase" in test and test["--testcase"] == 1:
                ranks += 1
        elif test["name"].lower() == "reference":
            if "--testcase" in test and test["--testcase"] == 1:
                checkIfParamExists(test, "-p", "--partition", "Invalid test: P not specified")
                ranks = test["-p"]
            else:
                checkIfParamExists(test, "-p1", "--partition1", "Invalid test: P1 not specified")
                checkIfParamExists(test, "-p2", "--partition2", "Invalid test: P2 not specified")
                ranks = test["-p1"] * test["-p2"]
        test["ranks"] = ranks
    else:
        test["-nx"] = size
        test["-ny"] = size
        test["-nz"] = size

    command = "mpiexec -n " + str(test["ranks"])
    if additional_flags != "":
        command += " " + additional_flags
    command += " " + test["name"].lower()

    for key in test:
        if key != "name" and key != "ranks":
            if type(test[key]) == type(True) and test[key] == True:
                command += " " + key 
            elif type(test[key]) != type(True):
                command += " " + key + " " + str(test[key])
    print(command)
    os.system(command)

if len(sys.argv) == 1:
    opt = menu_main()
    if opt == 0:
        jobs = select_job()
else:
    for i in range(1, len(sys.argv)):
        jobs.append("../jobs/" + sys.argv[i] + ".json")

hostname = os.uname()[1]
if hostname[0:6] == "pcsgs0":
    os.chdir("build_pcsgs")
elif hostname == "krypton":
    os.chdir("build_krypton")
else:
    os.chdir("build")
print()
if opt == 0:
    for job in jobs:
        filename = "../" + job
        with open(filename) as f:
            data = json.load(f)

        for s in data["size"]:
            print("Starting computation for size {}".format(s))
            count = 0
            for test in data["tests"]:
                print("-> Executing test {}".format(count))
                run_test(test, s, data["global_test_settings"], data["additional-flags"], s==data["size"][0])
                count += 1
                print()

elif opt == 1:
    print("todo")
else:
    print("todo")

    