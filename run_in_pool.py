import subprocess
import sys
import json
import os
from pathlib import Path

# This scripts runs the configurations given in the following 
# list of dictionaries
DATASET="Shuffle-2"
PYTHON_COMMAND="/itf-fi-ml/home/maihv/.conda/envs/transformers-fl/bin/python3"
EXTRA_PARAMS= " -gpu 0"
hyperparameters={
        "d_model": {"begin":2, "end":32},
        "depth": {"begin":1, "end":4},
        "heads": {"begin":1, "end":4}, 
        "lr": [0.01,0.001],
        "run_params": ["-model_type SAN", "-model_type SAN -pos_encode","-model_type SAN-Rel"]
        }
pending_directory = 'run_pool/pending'
running_directory = 'run_pool/running'
done_directory = 'run_pool/done'
results_directory = 'run_pool/results'
logs_directory = 'run_pool/logs'


def run_cmd(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT )
    out, err = p.communicate()
    print(out)
    print(err)
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


# iterate over files in pending
# that directory
there_is_item = True
while there_is_item:
    there_is_item = False
    files = Path(pending_directory).glob('*command')

    for file in files:
        f_name=str(file.name)
        task_name=f_name.replace(".command","")
        try:
            f=open(file,"r")
            command=f.readline().strip()
            f.close()
        except:
            there_is_item=True
            break

        command+=EXTRA_PARAMS
        running_file_name=running_directory + "/" + f_name
        done_file_name=done_directory + "/" + f_name
        log_file_name=logs_directory + "/" + task_name + ".log"

        try:
            log_file.close()
        except:
            pass

        log_file=open(log_file_name, "w")


        # Put the file into running directory so that other processes won't get it again
        try:
            os.replace(str(file), running_file_name)
        except:
            there_is_item=True
            break

        print("Processing: " , f_name)

        # Run the command
        log_file.write("Running: " + command)
        out, err, return_code = run_cmd(command.split(" "))
        out = str(out)
        err = str(err)

        result_string = ""
        print(out)
        print(err)
        f=open(results_directory + "/" + task_name + ".result","r")
        result_string=f.readline().strip()
        f.close()

        results_file=open(results_directory + "/" + task_name + ".result","w")
        results_file.write(result_string)
        results_file.write("\t")

        # If there is no proper output:
        if return_code!= 0:
            log_file.write(err)
            log_file.write("No output for the command : " + command)
            results_file.write("NA\tNA\n")
            results_file.close()
            os.replace(running_file_name, done_file_name)
            there_is_item=True
            break

        # Try to get the saved file name
        filename=""
        index1=out.find("Scores saved at ")
        if index1!= -1:
            index2=out.find("\\n", index1)
            filename=out[index1+16:index2].strip()

        # Read the file
        if filename!="":
            f = open(filename)
            data = json.load(f)
            data = data[task_name]
            gen=item_generator(data,"max_val_acc_bin0")

            num=0
            for i in gen:
                max_val_acc_bin0 = i
                num +=1
            if num != 1:
                log_file.write("There is no or multiple max_val_acc_bin0 values")
                results_file.write("NA\t")
                results_file.close()
                os.replace(running_file_name, done_file_name)
                there_is_item=True
                break

            results_file.write(str(max_val_acc_bin0) + "\t")

            gen=item_generator(data,"max_val_acc_bin1")
            num=0
            for i in gen:
                max_val_acc_bin1 = i
                num+=1
            if num != 1:
                log_file.write("There is no or multiple max_val_acc_bin1 values")
                results_file.write("NA\t")
                results_file.close()
                os.replace(running_file_name, done_file_name)
                there_is_item=True
                break

            results_file.write(str(max_val_acc_bin1) + "\t")

            gen=item_generator(data,"max_val_acc_bin2")
            num=0
            for i in gen:
                max_val_acc_bin2 = i
                num+=1
            if num != 1:
                log_file.write("There is no or multiple max_val_acc_bin2 values")
                results_file.write("NA\n")
                results_file.close()
                os.replace(running_file_name, done_file_name)
                there_is_item=True
                break

            results_file.write(str(max_val_acc_bin2) + "\n")


        else:
            print(err)
            log_file.write("No output for the command : " + command)
            there_is_item=True
            results_file.close()
            os.replace(running_file_name, done_file_name)
            break

        results_file.close()
        os.replace(running_file_name, done_file_name)
        there_is_item=True

        # Breaking here for the upper while loop to get another file name
        break

