import subprocess
import sys
import json
from pathlib import Path
import os

# This scripts runs the configurations given in the following 
# list of dictionaries
DATASET="SL_2_3_1_u-bbb"
PYTHON_COMMAND="/itf-fi-ml/home/maihv/.conda/envs/transformers-fl/bin/python3"
RUN_POOL_DIR="./run_pool"
PENDING_DIR=RUN_POOL_DIR + "/pending"
DONE_DIR=RUN_POOL_DIR + "/done"
LOGS_DIR=RUN_POOL_DIR + "/logs"
RESULTS_DIR=RUN_POOL_DIR + "/results"  
RUNNING_DIR=RUN_POOL_DIR + "/running"


hyperparameters={
        "d_model": {"begin":2, "end":32},
        "depth": {"begin":1, "end":2},
        "heads": {"begin":1, "end":4}, 
        "lr": [0.01,0.001],
        "run_params": ["-model_type SAN","-model_type SAN -pos_encode"]
        }

def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    try:
        pth.rmdir()
    except:
        pass


def run_cmd(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT )
    out, err = p.communicate()
    return (out,err, p.returncode)

# Find the specific key's value recursively
# From : https://stackoverflow.com/questions/21028979/how-to-recursively-find-specific-key-in-nested-json
def item_generator(json_input, lookup_key):
    if isinstance(json_input, dict):
        for k, v in json_input.items():
            if k == lookup_key:
                yield v
            else:
                yield from item_generator(v, lookup_key)
    elif isinstance(json_input, list):
        for item in json_input:
            yield from item_generator(item, lookup_key)


def combine(elems):
    if len(elems) == 0:
        return [[]]
    return [[x, *y] for x in elems[0] for y in combine(elems[1:])]


# Remove the pool directory and recreate
rm_tree(RUN_POOL_DIR)
os.mkdir(RUN_POOL_DIR)
os.mkdir(PENDING_DIR)
os.mkdir(DONE_DIR)
os.mkdir(LOGS_DIR)
os.mkdir(RESULTS_DIR)
os.mkdir(RUNNING_DIR)




results_file = open (RUN_POOL_DIR+"/head.tsv", "w")
for par in hyperparameters:
    results_file.write(par + "\t")
results_file.write("max_val_acc_bin0\tmax_val_acc_bin1\tmax_val_acc_bin2\n")
results_file.close()

# Iterate over hyperparameter combinations
iteration_dict = {}
for par in hyperparameters:
    item = hyperparameters[par]
    
    if type(item) == dict:
        if "begin" in item and "end" in item:
            iteration_dict[par] =[par+"$$" +str(i) for i in list(range(item["begin"], item["end"]+1))]

    elif type(item) == list:
        iteration_dict[par]=[ par+"$$"+str(i) for i in item]
iteration_list=[]
for par in iteration_dict:
    iteration_list.append(iteration_dict[par])

combinations = combine(iteration_list)
for combination in combinations:
    command = PYTHON_COMMAND+" -m src.main -mode train -dataset " + DATASET
    name =""
    config=dict()
    for par in combination:
        par = par.split("$$")
        if par[0]=="run_params":
            name += " " + par[1]
        else:
            name += " -" + par[0] + " " + par[1]
        config[par[0]]=par[1]

    run_name = name.strip().replace(" ", "_").replace("-","").replace("_run_params","")
    command += " -epochs 100 -run_name " + run_name + name 
    f=open(PENDING_DIR + "/" + run_name + ".command", "w")
    f.write(command)
    f.close()

    f=open(RESULTS_DIR + "/" + run_name + ".result", "w")
    for par in hyperparameters:
        f.write(config[par] + "\t")
    f.close()
