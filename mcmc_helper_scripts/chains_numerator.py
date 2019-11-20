#!/usr/bin/env python

# Chains Numerator
# ---
# Abhimat Gautam

import numpy as np
from tqdm import tqdm
import h5py
from emcee import backends

try_num = '1'
out_file = open('chains_samples_try{0}.txt'.format(try_num), 'w')

header_line = '{0:<35} {1:<15} {2:<20} {3:<20}\n'.format('Test',
                                                         'Num Chains', 'Num Steps/Chain',
                                                         'Params Calculated')
out_file.write(header_line)

# Get all subdirectories in current directory
from glob import glob
import os

sub_dirs = glob("./*/")

for cur_sub_dir in tqdm(np.sort(sub_dirs)):
    chains_file = '{0}chains/chains_try{1}.h5'.format(cur_sub_dir, try_num)
    params_file = '{0}stellar_params/stellar_params.h5'.format(cur_sub_dir)
    
    test_name = cur_sub_dir.strip('./')
    
    if os.path.exists(chains_file):
        chains_reader = backends.HDFBackend(chains_file, read_only=True)
        samples = chains_reader.get_chain()
        (num_steps, num_chains, num_params) = samples.shape
        
        num_samples_params = 0
        
        if os.path.exists(params_file):
            f = h5py.File(params_file, 'r')
            total_rows, = f['data'].shape
            f.close()
            
            num_samples_params = total_rows / num_chains
        
        out_line = '{0:<35} {1:<15} {2:<20} {3:<20}\n'.format(test_name,
                                                              num_chains, num_steps,
                                                              num_samples_params)
        out_file.write(out_line)
    else:
        out_line = '{0:<35} {1:<15} {2:<20} {3:<20}\n'.format(test_name, '--', '--', '--')
        out_file.write(out_line)
    

out_file.close()
