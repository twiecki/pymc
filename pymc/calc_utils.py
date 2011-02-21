'''
Created on Jan 20, 2011

@author: jsalvatier
'''
import numpy as np

def sum_to_shape(value, sum_shape):
    
    value_shape = np.array(np.shape(value))
    
    sum_shape_expanded = np.zeros(value_shape.size)
    sum_shape_expanded[0:len(sum_shape)] += np.array(sum_shape)
    
    axes = np.where(sum_shape_expanded != value_shape)[0]
    lx = np.size(axes)
     
    if lx > 0:
        return np.apply_over_axes(np.sum, value, axes)

    else:
        return value