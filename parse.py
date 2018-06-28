#!/usr/bin/env python

# pylint: disable=R0915
# pylint: disable=R0914

"""Parse the full_dictionary into a bunch of smaller ones, and generate all
input SLURM and bash scripts."""

__author__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"

import pickle
import os 
from math import floor, log10
from itertools import product

# local imports
from harmony.utils import all_parameters

def main():
    """Parses a given dictionary and generates necessary scripts for SLURM
    submission."""

    [slurm, __, full_dictionary] = all_parameters()

    acct = slurm['ACCT']
    partition = slurm['PARTITION']
    mail = slurm['MAIL']
    procs_per_node = slurm['PROCS_PER_NODE']
    sort_by = slurm['SORT_BY']
    gpu = slurm['GPU']

    # dictionary checker: assert no critical conflicts
    # unnecessary for now given how all_parameters() works
    # assert_hyperparameters(full_dictionary)

    # split the dictionary
    combinations = [dict(zip(full_dictionary, prod))
                    for prod in product(*(full_dictionary[ii] 
                                          for ii in full_dictionary))]
    n_combo = len(combinations)
    order_mag = int(floor(log10(n_combo)))

    # ensure the combinations are sorted by epoch (or alternatively, change
    # the sort_by constant to sort by a different key value - not recommended)
    combinations = sorted(combinations, key=lambda k: k[sort_by]) 

    # write the sbatch SLURM scripts
    try:
        os.makedirs('job_data')
    except FileExistsError:
        pass

    # make a submit all script
    sub_all = open('submit_all.sh', 'w')
    sub_all.write("#!/bin/bash\n")
    sub_all.write('module load anaconda3\n')

    # first, assert that the total number of combinations is modulo the
    # number of cpus per node
    # only going to submit efficiently to the cluster
    if n_combo % procs_per_node != 0:
        raise RuntimeError("N.o. combinations (%i) must be divisible by "
                           "the number of cpus/node (%i) " 
                           % (n_combo, procs_per_node))

    outer_limit = int(n_combo / procs_per_node)
    order_mag_outer_limit = int(floor(log10(outer_limit)))

    for outer_i in range(outer_limit):

        outer_i_string = str(outer_i+1).zfill(order_mag_outer_limit+1)

        file = open('submit%s.sbatch' % outer_i_string, 'w')

        sub_all.write('sbatch submit%s.sbatch\n' % outer_i_string)

        file.write("#!/bin/bash\n")
        file.write("#SBATCH --account=%s\n" % acct)
        file.write("#SBATCH --nodes=1\n")
        file.write("#SBATCH --partition=%s\n" % partition)
        file.write("#SBATCH --time=02:00:00\n")
        file.write("#SBATCH --job-name=ht%s\n" % outer_i_string)
        file.write("#SBATCH --mail-user=%s\n" % mail)
        file.write("#SBATCH --mail-type=ALL\n")
        file.write("#SBATCH --output=job_data/ht_%sA.out\n" % '%')
        file.write("#SBATCH --error=job_data/ht_%sA.err\n" % '%')
        file.write("\n")

        file.write("SECONDS=0\n") 
        file.write("module load anaconda3\n")

        if not gpu:
            file.write("export CUDA_VISIBLE_DEVICES=''\n")

        for inner_i in range(procs_per_node):
            task_id = procs_per_node*outer_i + inner_i + 1

            file.write("OMP_NUM_THREADS=1 ./ex.py %i %i &\n" 
                       % (task_id, order_mag))

        file.write("wait\n")
        file.write("echo %s: $SECONDS" % outer_i_string)
        file.close()
    sub_all.close()

    try:
        os.makedirs('train')
    except FileExistsError:
        # directory already exists
        pass

    for i in range(n_combo):
        i_str = str(i+1).zfill(order_mag+1)
        try:
            os.makedirs('train/' + i_str)
        except FileExistsError:
            pass

        # now save the corresponding mini-dictionaries to their directories
        with open('train/%s/d.pkl' % i_str, 'wb') as file:
            pickle.dump(combinations[i], file)
        file.close()

        # write the dictionary to a text file for human use
        with open('train/%s/d.txt' % i_str, 'w') as file:
            for j in combinations[i]:
                file.write("%s : %s\n" % (j, combinations[i][j]))
        file.close()


if __name__ == '__main__':
    main()
