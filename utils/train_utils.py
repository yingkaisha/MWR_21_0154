import glob
import random
import numba as nb
import numpy as np

import sys
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
from data_utils import *

def random_cropping_single(input3d, input2d, land_mask, size, gap, ocean_f=1/2, sparse_f=1/2, shift=4):
    '''
    random cropping
    '''
    thres = 0.0
    
    if shift==0:
        flag_rnd = False
    else:
        flag_rnd = True
        
    if sparse_f == 0:
        flag_s = False
    else:
        flag_s = True
    
    Nvar_3d = len(input3d)
    Nvar_2d = len(input2d)
    
    # cropping detection
    Lx, Ly = land_mask.shape # domain size
    Nx = (Lx-size+1)//gap+1  # derive number of cropping by domain size
    Ny = (Ly-size+1)//gap+1

    start_flag = np.zeros((Nx, Ny))
    # check if croppings can match "ocean_f"
    for i in range(Nx):
        for j in range(Ny):
            N_ocean_grid = np.sum(land_mask[gap*i:gap*i+size, gap*j:gap*j+size])
            if N_ocean_grid<=ocean_f*size*size:
                start_flag[i, j] = True
            else:
                start_flag[i, j] = False    
    
    # allocating the output array
    N_single = np.sum(start_flag) # largest possible samples in one day
    N_total = int(N_single)
    CROPPINGs = np.empty((N_total, size, size, Nvar_2d+Nvar_3d)) 

    # exact number of samples
    count = 0 

    # loop of x-indices of croppings
    for indx in range(Nx):
        # loop of y-indices of croppings
        for indy in range(Ny):
            # if the cropping satisfies ocean_f 
            if start_flag[indx, indy]:
                ind_xs = gap*indx; ind_xe = gap*indx+size
                ind_ys = gap*indy; ind_ye = gap*indy+size
                # adding random shifts <--- cannot larger than "gap"
                if flag_rnd:
                    if ind_xs > shift and ind_xe < Lx-shift:
                        ind_xs += int(shift); ind_xe += int(shift)
                    if ind_ys > shift and ind_ye < Ly-shift:
                        ind_ys += int(shift); ind_ye += int(shift)

                test_frame = input3d[0][ind_xs:ind_xe, ind_ys:ind_ye]
                # if the cropping satisfies sparse_f
                if flag_s:
                    N_zeros = np.sum(np.logical_or(np.isnan(test_frame), test_frame<=thres))
                    if N_zeros >= sparse_f*size*size:
                        continue;
                if test_frame.shape == (size, size):
                    # selected input features
                    for m in range(Nvar_3d):
                        CROPPINGs[count, ..., m] = input3d[m][ind_xs:ind_xe, ind_ys:ind_ye]
                    for m in range(Nvar_2d):
                        CROPPINGs[count, ..., m+Nvar_3d] = input2d[m][ind_xs:ind_xe, ind_ys:ind_ye]
                    count += 1
    return CROPPINGs[:count, ...], count


#@nb.njit(fastmath=True)
def random_cropping(input3d, input2d, land_mask, size, gap, ocean_f=1/2, sparse_f=1/2, rnd_range=4):
    '''
    random cropping
    '''
    thres = 0.0
    
    if rnd_range==0:
        flag_rnd = False
    else:
        flag_rnd = True
    if sparse_f == 0:
        flag_s = False
    else:
        flag_s = True
    
    Nvar_3d = len(input3d)
    Nvar_2d = len(input2d)
    L_all = len(input3d[0])
    
    # cropping detection
    Lx, Ly = land_mask.shape # domain size
    Nx = (Lx-size+1)//gap+1  # derive number of cropping by domain size
    Ny = (Ly-size+1)//gap+1

    start_flag = np.zeros((Nx, Ny))
    # check if croppings can match "ocean_f"
    for i in range(Nx):
        for j in range(Ny):
            N_ocean_grid = np.sum(land_mask[gap*i:gap*i+size, gap*j:gap*j+size])
            if N_ocean_grid<=ocean_f*size*size:
                start_flag[i, j] = True
            else:
                start_flag[i, j] = False    
    
    # allocating the output array
    N_single = np.sum(start_flag) # largest possible samples in one day
    N_total = int(N_single*L_all)     # largest possible samples in all days
    CROPPINGs = np.empty((N_total, size, size, Nvar_2d+Nvar_3d)) 

    # exact number of samples
    count = 0 
    # loop of time
    for i in range(L_all):
        # loop of x-indices of croppings
        for indx in range(Nx):
            # loop of y-indices of croppings
            for indy in range(Ny):
                # if the cropping satisfies ocean_f 
                if start_flag[indx, indy]:
                    ind_xs = gap*indx; ind_xe = gap*indx+size
                    ind_ys = gap*indy; ind_ye = gap*indy+size
                    # adding random shifts <--- cannot larger than "gap"
                    if flag_rnd:
                        if ind_xs > rnd_range and ind_xe < Lx-rnd_range:
                            d = np.random.randint(-1*rnd_range, rnd_range)
                            ind_xs += int(d); ind_xe += int(d)
                        if ind_ys > rnd_range and ind_ye < Ly-rnd_range:
                            d = np.random.randint(-1*rnd_range, rnd_range)
                            ind_ys += int(d); ind_ye += int(d)

                    test_frame = input3d[0][i, ind_xs:ind_xe, ind_ys:ind_ye]
                    # if the cropping satisfies sparse_f
                    if flag_s:
                        N_zeros = np.sum(np.logical_or(np.isnan(test_frame), test_frame<=thres))
                        if N_zeros >= sparse_f*size*size:
                            continue;
                    if test_frame.shape == (size, size):
                        # selected input features
                        for m in range(Nvar_3d):
                            CROPPINGs[count, ..., m] = input3d[m][i, ind_xs:ind_xe, ind_ys:ind_ye]
                        for m in range(Nvar_2d):
                            CROPPINGs[count, ..., m+Nvar_3d] = input2d[m][ind_xs:ind_xe, ind_ys:ind_ye]
                        count += 1
    return CROPPINGs[:count, ...]

def feature_norm_(FEATURES, ):
    shape_ = FEATURES.shape
    for i in range(shape_[0]):
        FEATURES[i, ..., :-1] = np.log(FEATURES[i, ..., :-1]+1)

    FEATURES[np.isnan(FEATURES)] = 0
    return FEATURES

def batch_gen(FEATUREs, batch_size, BATCH_dir, perfix, ind0=0):
    '''
    Spliting and generating batches as naive numpy files
        - perfix: filename
        - ind0: the start of batch index
        e.g., 'perfix0.npy' given ind0=0
    '''
    L = len(FEATUREs)
    N_batch = L//batch_size # losing some samples
    
    #print('\tNumber of batches: {}'.format(N_batch))
    # shuffle
    ind = shuffle_ind(L)
    FEATUREs = FEATUREs[ind, ...]
    # loop
    for i in range(N_batch):
        save_d = FEATUREs[batch_size*i:batch_size*(i+1), ...]
        temp_name = BATCH_dir+perfix+str(ind0+i)+'.npy'
        print(temp_name) # print out saved filenames
        np.save(temp_name, save_d)
    return N_batch
