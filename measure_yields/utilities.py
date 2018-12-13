# pylint: disable=C0303

"""
"""

import os, sys
import numpy as np
from collections import namedtuple
#from scipy.stats import norm
import h5py
from glob import glob
import pickle
#from keras.models import load_model
#from sklearn.preprocessing import binarize

try:
    import matplotlib.pyplot as plt
    disable_graphics = False
except ImportError:
    disable_graphics = True

from contextlib import contextmanager
import ROOT
from array import array
from copy import deepcopy







def data_already_scaled(data):
    """
    Test if data looks like it is already standardised
    """
    return abs(np.mean(np.mean(data, axis=0))) < 0.1




def get_adapted_bin_edges(data, fullrange):
    """
    Set appropriate bin edges for the data, so that the number of non-empty
    bins is always the same
    
    Arguments:
        data: Vector of data
        fullrange: Full range of the observable (tuple)
        nfilledbins: Number of bins required to be filled
    """
    
    percent_of_data_per_bin = 1.0

    required_entries_per_bin = float(len(data))*percent_of_data_per_bin*0.01
    print 'required_entries_per_bin =', required_entries_per_bin
    
    
    # Initial histogram
    contents, edges = np.histogram(data, bins=100, range=fullrange)
    print 'hist:', contents, edges
    
    # Start at the left side, move right towards the peak
    peakbin = np.argmax(contents)
    
    peakval = contents[peakbin]
    
    i = peakbin
    edges_right = []
    while i < len(contents):
        if contents[i] > 0.5*peakval: 
            print 'contents[%d] = %d, continue' % (i, contents[i])
            edges_right.append(edges[i+1])
        
        else:
            print 'contents[%d] = %d ' % (i, contents[i])
            if contents[i] < 1 and i < len(contents)-1:
                # Proceed to next filled bin
                jump = 0
                j = i
                while contents[j] == 0:
                    if j+1 >= len(contents):
                        edges_right.append(edges[-1])
                        print ' ohoi'
                        break
                    jump += 1
                    j += 1
                    print '   trying jump =', jump, 'contents = ', contents[j]
            elif i == len(contents)-1:
                print 'hoi'
                edges_right.append(edges[-1])
                break
            else:
                jump = int(1.0/(contents[i]/float(peakval)))
            print 'jump =', jump
            i += jump
            if i > len(contents)-1:
                edges_right.append(edges[-1])
                break
            edges_right.append(edges[i])
        print 'edges_right =', edges_right
        
        i += 1
        #raw_input('cont')
    print 'final edges_right =', edges_right
    
    i = peakbin
    edges_left = []
    while i >= 0:
        if contents[i] > 0.5*peakval: 
            print 'contents[%d] = %d, continue' % (i, contents[i])
            edges_left.append(edges[i])
        
        else:
            print 'contents[%d] = %d ' % (i, contents[i])
            if contents[i] < 1 and i >= 0:
                # Proceed to next filled bin
                jump = 0
                j = i
                while contents[j] == 0:
                    if j < 1:
                        edges_left.append(edges[0])
                        print ' ohoi'
                        break
                    jump += 1
                    j -= 1
                    print '   trying jump =', jump, 'contents = ', contents[j]
            elif i == 0:
                print 'hoi'
                edges_left.append(edges[0])
                break
            else:
                jump = int(1.0/(contents[i]/float(peakval)))
            print 'jump =', jump
            i -= jump
            if i < 0:
                edges_left.append(edges[0])
                break
            edges_left.append(edges[i])
        print 'edges_left =', edges_left
        
        i -= 1
    print 'final edges_left =', edges_left

    newedges = sorted(edges_left + edges_right)
    print 'newedges:', newedges
    print 'new hist:', np.histogram(data, newedges)

    return newedges



if __name__ == '__main__':
    
    # a = np.random.normal(loc=0.5, scale=0.1, size=10)
    a = [ 0.52387577,  0.5550518 ,  0.45878739,  0.64108826,  0.4220505 ,
        0.49375241,  0.58149284,  0.73435625,  0.61300805,  0.52538784]
    get_adapted_bin_edges(a, (0.,1.))
    
    
        
