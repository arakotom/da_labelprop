#%%
import subprocess
import os
import argparse
import sys

def get_specific_command(data, bag_size):
    spec = ""

    return spec

sys.argv = ['']
parser = argparse.ArgumentParser(description="train model")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
seed = args.seed
current_path = os.getcwd()
data = 'office31'
list_problems = [0]
algo_list = ['bagCSI','daLabelWD']
algo_list = ['bagCSI']
n_param = 1
list_bag_size = [50]
for i_p in range(n_param):
    for bag_size in list_bag_size:
        for problem in list_problems:
            for algo in algo_list:
                specific_command = get_specific_command(data, bag_size)


                if "a.rakoto" in current_path:
                    command = f"nohup  python -u expe.py  --data {data:}  --algo {algo:} --seed {seed:} "
                    command += f"--bag_size {bag_size:} --source_target {problem:} --i_param {i_p:} "
                    command += specific_command
                    command += f"  > ./out/out_bs_{data}_{algo}_{problem}_{bag_size}_{seed}.log"

                    os.system(command)
                else:
                    os.system("rm slurm*.out")
                    print('sbatch tasks')
                    command = f"sbatch expe.slurm {data:} {algo:} {bag_size} {problem:} {i_p:} {seed:} &"
                    os.system(command)

# %%
