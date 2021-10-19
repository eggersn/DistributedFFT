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


import json
import os, sys
from os import listdir
from os.path import isfile, join, isdir
import argparse
import multiprocessing
import re
import subprocess
from datetime import datetime

def menu_main():
    valid = False
    os.system("clear")
    while not valid:
        print("Select a Category:\n" +
        "-" * 35 + "\n" + 
        "[0] Run Specified Job (job.json)\n" +
        "[1] Run Evaluation Scripts \n\n" +
        "Selection: ")
        try: 
            opt = int(input())
            if opt >= 0 and opt <= 1:
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
    prefix = "../jobs"
    while True:
        os.system("clear")
        valid = False
        jobs = [f for f in listdir(prefix)]
        while not valid:
            print("Select a Job:\n" + "-" * 35)
            print("[0] all")
            for i in range(len(jobs)):
                if jobs[i] != "slurm_scripts" and jobs[i] != "launch_scripts":
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

def select_eval():
    valid = False
    script_collection = ["eval/global_redist/evaluation_slab.py", "eval/global_redist/evaluation_pencil.py", "eval/global_redist/evaluation_single.py", "eval/complete/plot_complete.py", "eval/complete/numerical_results.py"]
    os.system("clear")
    while not valid:
        print("Select the Evaluation Scripts:\n" +
        "-" * 35 + "\n" + 
        "[0] All (equal to [2] + [3] + [4] -> [1])\n" +
        "[1] Summary (requires [2] and/or [3], ) \n" +
        "[2] Slab Decomposition\n" +
        "[3] Pencil Decomposition\n" +
        "[4] Single\n" +
        "Selection: ")
        try: 
            opt = int(input())
            if opt >= 2 and opt <= 4:
                return [script_collection[opt-2]]
            elif opt == 1:
                return [script_collection[3], script_collection[4]]
            elif opt == 0:
                return script_collection
        except ValueError: 
            os.system("clear")
            print("Please enter an integer!\n")
        except KeyboardInterrupt:
            print("\nexiting...")
            sys.exit()

    return 0

def rec_benchmarks(prefix):
    if isdir(join(prefix, "forward")) or isdir(join(prefix, "inverse")):
        return [prefix] 

    subdir = [f for f in listdir(prefix)]
    res = []
    for d in subdir:
        res += rec_benchmarks(join(prefix, d))
    return res


def select_benchmark():
    selected_benchmarks = []
    prefix = "benchmarks"
    while True:
        os.system("clear")
        valid = False
        benchmarks = [f for f in listdir(prefix)]
        while not valid:
            print("Select a Benchmark:\n" + "-" * 35)
            print("[0] all")
            for i in range(len(benchmarks)):
                print("[{}] {}".format(i+1, benchmarks[i]))
            print("Selection: ")
            try: 
                opt = int(input())
                if opt == 0:
                    # find all valid subdirs of benchmarks
                    return rec_benchmarks(prefix)
                if opt > 0 and opt <= len(benchmarks):
                    if isdir(join(prefix, benchmarks[opt-1], "forward")) or isdir(join(prefix, benchmarks[opt-1], "inverse")):
                        return [prefix + "/" + benchmarks[opt-1]]
                    prefix += "/" + benchmarks[opt-1]
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
        del test[key2]

def checkIfParamExists(test, key1, key2, error_msg=""):
    if (not key1 in test) and (not key2 in test):
        raise LookupError(error_msg)
    convertKey(test, key1, key2)

def run_test(test, size, global_test_settings, additional_flags, parse):
    if parse == True:
        if "--cuda_aware" in test and test["--cuda_aware"] == False:
            test.update(global_test_settings)
            test["--cuda_aware"] = False
        else:    
            test.update(global_test_settings)
        if size != 0:
            if "--input-dim-x" in test or "-nx" in test:
                print("Warning: Nx is overridden by size!")
            if "--input-dim-y" in test or "-ny" in test:
                print("Warning: Ny is overridden by size!")
            if "--input-dim-z" in test or "-nz" in test:
                print("Warning: Nz is overridden by size!")

            if type(size) == list:
                test["-nx"] = size[0]
                test["-ny"] = size[1]
                test["-nz"] = size[2]
            else:
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
            ranks = int(test["-p1"]) * int(test["-p2"])
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
        if type(size) == list:
            test["-nx"] = size[0]
            test["-ny"] = size[1]
            test["-nz"] = size[2]
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
    print(datetime.now())
    try:
        output = subprocess.check_output(command, shell=True)
        print(output)
    except subprocess.CalledProcessError as e:
        print(e.output)

def generateHostfile(hosts, id=0):
    # assume same hardware for all workers
    cpus = multiprocessing.cpu_count()

    with open("../mpi/hostfile_{}".format(id), "w") as f:
        for host in hosts:
            f.write("{}\tslots={}\n".format(host, cpus))
        f.close()
    return "../mpi/hostfile_{}".format(id)

def generateRankfile(hosts, gpus, affinity, id=0):
    rank = 0
    with open("../mpi/rankfile_{}".format(id), "w") as f:
        for host in hosts:
            for gpu in range(0, gpus):
                f.write("rank {}={} \tslot={}\n".format(rank, host, affinity[gpu]))
                rank += 1
        f.close()
    return "../mpi/rankfile_{}".format(id)

def main():
    parser = argparse.ArgumentParser(description='Launch script for the performance benchmarks.')
    parser.add_argument('--jobs', metavar="j1", type=str, nargs='+', dest='jobs', help='A list of jobs (located in the ./jobs folder), where individual jobs are seperated by spaces. Example \"--jobs home/slab/zy_then_x.json home/slab/z_then_yx.json\"')
    parser.add_argument('--global_params', metavar="p", type=str, nargs=1, dest="global_params", help="Params passed to slab, pencil or reference MPI call.")
    parser.add_argument('--mpi_params', metavar="p", type=str, nargs=1, dest="mpi_params", help="Params passed to MPI.")
    parser.add_argument('--hosts', metavar="h1", type=str, nargs='+', dest='hosts', help='A list of hostnames seperated by spaces, specifying which hosts to use for MPI execution.')
    parser.add_argument('--id', metavar="n", type=str, nargs=1, dest="id", help="Identifier for host- and rankfile in the ./mpi folder. Is required for parallel execution of tests (using this script) to avoid ambiguity.")
    parser.add_argument('--gpus', metavar="n", type=int, nargs=1, dest="gpus", help="Number of GPUs per node.")
    parser.add_argument('--affinity', metavar="c", type=str, nargs='+', dest="affinity", help="List of cores for GPU to bind to. The list has to be of length --gpus. Example: \"--affinity 0:0-9 1:20-29\". Here the first rank is assinged to cores 0-9 on socket 0 for GPU0 and the second rank is assinged to cores 20-29 on socket 1 for GPU1.")
    parser.add_argument('--build_dir', metavar="d", type=str, nargs=1, dest="build_dir", help="Path to build directory (default: ./build).")

    args = parser.parse_args()
    hostfile = ""
    rankfile = ""

    if args.build_dir != None:
        os.chdir(args.build_dir[0])
    else:
        os.chdir("build")

    if args.hosts != None:
        args.hosts = list(dict.fromkeys(args.hosts))
        if args.id != None:
            hostfile = generateHostfile(args.hosts, args.id[0])
        else:
            hostfile = generateHostfile(args.hosts)

        if (args.gpus != None) and (args.affinity != None) and (args.gpus[0] == len(args.affinity)):
            if args.id != None:
                rankfile = generateRankfile(args.hosts, args.gpus[0], args.affinity, args.id[0])
            else:
                rankfile = generateRankfile(args.hosts, args.gpus[0], args.affinity)
        elif (args.gpus != None) or (args.affinity != None):
            raise ValueError('Arguments --gpus and --affinity do not fit together! See --help for more information.')

    opt = 0
    if args.jobs == None:
        opt = menu_main()
        if opt == 0:
            jobs = select_job()
        else:
            os.chdir("..")
            eval_scripts = select_eval() 
            benchmarks = select_benchmark()
            
            for script in eval_scripts:
                for benchmark in benchmarks:
                    os.system("python {} --prefix {}".format(script, benchmark))
    else:
        jobs = ["../jobs/" + job for job in args.jobs]

        
    if opt == 0:
        for job in jobs:
            filename = job
            with open(filename) as f:
                data = json.load(f)

            if args.global_params != None:
                program_args = args.global_params[0].split(" ")
                i = 0
                while i < len(program_args):
                    if program_args[i] == "-c" or program_args[i] == "--cuda_aware":
                        data["global_test_settings"]["--cuda_aware"] = True
                        i += 1
                    elif program_args[i] == "-d" or program_args[i] == "--double_prec":
                        data["global_test_settings"]["--double_prec"] = True
                        i += 1
                    elif len(program_args[i]) > 0 and i+1 < len(program_args):
                        data["global_test_settings"][program_args[i]] = program_args[i+1]
                        i += 2
                    else:
                        i += 1

            old_keys = list(data["global_test_settings"].keys())[:]
            for key in old_keys:
                if key[0] == "$":
                    data["global_test_settings"][key[1:]] = data["global_test_settings"][key]
                    del data["global_test_settings"][key]

            if args.mpi_params != None:
                data["additional-flags"] = args.mpi_params[0]

            if hostfile != "" and re.match("--hostfile", data["additional-flags"]):
                raise ValueError('Error in job {}: Hostfile is specified even though it is newly generated, aborting...'.format(job))
            elif hostfile != "":
                data["additional-flags"] += " " + "--hostfile {}".format(hostfile)
            if rankfile != "" and re.match("--rankfile", data["additional-flags"]):
                raise ValueError('Error in job {}: Rankfile is specified even though it is newly generated, aborting...'.format(job))
            elif rankfile != "":
                data["additional-flags"] += " " + "--rankfile {}".format(rankfile)

            for s in data["size"]:
                print("Starting computation for size {}".format(s))
                count = 0
                for test in data["tests"]:
                    print("-> Executing test {}".format(count))
                    run_test(test, s, data["global_test_settings"], data["additional-flags"], s==data["size"][0])
                    count += 1
                    print()


if __name__ == "__main__":
    main()