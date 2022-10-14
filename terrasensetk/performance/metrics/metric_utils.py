import pandas as pd
import numpy as np


def aux_get_size(state_gt):
    '''
    Gets the size of list/array for iteration
    '''

    temp_length = 0
    if isinstance(state_gt, list):
        temp_length = len(state_gt)
    else:
        temp_length = state_gt.shape[0]

    return temp_length


def aux_error_checking(state_gt, state_pred):
    '''
    Checks for list/array size incompatibility
    '''

    if isinstance(state_gt, list):
        if len(state_gt) != len(state_pred):
            print('Ground truth and predicted arrays must be of the same size')
            return True
        else:
            return False
    elif state_gt.shape[0] != state_pred.shape[0]:
        print('Ground truth and predicted arrays must be of the same size')
        return True
    else:
        return False