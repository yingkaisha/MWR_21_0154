
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def cnn_precip_fix(data):
    '''
    Precipitation values predicted by the CNNs are
    not non-negative. Fixing this issue with logical_and.
    '''
    
    flag_0 = data<0.5
    flag_1 = np.logical_and(data<0.6, data>=0.5)
    flag_2 = np.logical_and(data<0.7, data>=0.6)
    flag_3 = np.logical_and(data<0.8, data>=0.7)
    flag_4 = np.logical_and(data<0.9, data>=0.8)

    data[flag_0] = 0.0
    data[flag_1] = 0.1
    data[flag_2] = 0.3
    data[flag_3] = 0.5
    data[flag_4] = 0.7
    data[data>40] = 40
    return data





