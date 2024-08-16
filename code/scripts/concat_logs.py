import os
import sys
from glob import glob

from tqdm import tqdm

path = "./"
files = glob(os.path.join(path, "*.log")) + glob(os.path.join(path, "*.py")) + glob(os.path.join(path, "*.sh"))
output = os.path.join(path, "output.md")

def write_to_file(file, output):
    with open(file, "r", encoding="ISO-8859-1") as f:
        # basic information of the file:

        output.write(f"## {os.path.split(file)[-1]}\n")
        if file.endswith(".py"):
            output.write("```python\n")
        elif file.endswith(".sh"):
            output.write("```bash\n")
        else:
            output.write("```\n")

        # content of the file:
        for line in f:
            # filter tqdm outputs
            if "it/s" in line or "iters/s" in line or "s/it" in line:
                continue
            # filter training logs
            if "Epoch" in line or "Iteration" in line or "Loss" in line or "Acc" in line:
                continue
            # filter vina outputs
            """
            Detected 32 CPUs
            WARNING: at low exhaustiveness, it may be impossible to utilize all CPUs
            Reading input ... done.
            Setting up the scoring function ... *done.
            *Analyzing the binding site ... ****done.
            Using random seed: -121913038
            Performing search ... 
            0%   10   20   30   40   50   60   70   80   90   100%
            |----|----|----|----|----|----|----|----|----|----|
            *****
            done.
            Refining results ... **done.
            """
            if line.startswith("#") or "Detected" in line or "WARNING" in line or "Reading input" in line or "Setting up the scoring function" in line or "Analyzing the binding site" in line or "Using random seed" in line or "Performing search" in line or "Refining results" in line:
                continue
            """
            1 molecule converted
            Processing split 2648
            """
            if "molecule converted" in line or "Processing split" in line:
                continue
            output.write(line)

        output.write("```\n\n")

with open(output, "w") as output:
    for file in tqdm(files):
        if os.path.split(file)[-1] == "dock.log":
            continue
        write_to_file(file, output)

