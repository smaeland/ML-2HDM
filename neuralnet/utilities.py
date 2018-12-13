# Various functions common for the network training

import numpy as np
import h5py
from glob import glob


def select_features(data, feature_names_in_data, include_mass_variables=True):
    """
    Select certain columns from a data vector, corresponding to 4-vector
    components, IP vectors, phistar, etc. Returns a new numpy array
        
        feature_names_in_data: List of the names of the columns in 'data'
        include_mass_variables: Also return mass variables
    """        

    # Basic features
    features_to_use = ['piplus.px', 'piplus.pz', 'piplus.e',
                       'pi0plus.pz', 'pi0plus.e',
                       'piminus.px', 'piminus.py', 'piminus.pz', 'piminus.e',
                       'pi0minus.pz', 'pi0minus.e',
                       'met_x', 'met_y',
                       'upsilon_plus', 'upsilon_minus',
                       'triple_corr', 'phistar',
                       'piplus_ip_x', 'piplus_ip_y', 'piplus_ip_z',
                       'piminus_ip_x', 'piminus_ip_y', 'piminus_ip_z',
                      ]

    # Add masses
    if include_mass_variables:
        features_to_use.append('inv_mass')
        features_to_use.append('transv_mass')

    features_to_use = np.array(features_to_use)

    cols = []
    for name in features_to_use:
        if name in feature_names_in_data:
            cols.append(np.where(feature_names_in_data == name)[0][0])

    x = data[:, cols]

    return x, features_to_use



def get_dataset_from_path(path, pattern, include_mass=True):
    """ Open h5 files in path, matching name pattern. Return numpy arrays """
    
    X = np.array([])
    Y = np.array([])
    features = None

    if not path.endswith('/'):
        path += '/'
    
    filelist = glob(path + '*%s*.h5' % pattern)
    
    if len(filelist) < 1:
        print 'Error: no files found at', path, 'matching', pattern
        exit(-1)
    
    return get_dataset_from_files(filelist, include_mass)



def get_dataset_from_files(files, include_mass=True):
    """
    Open specific h5 files in a list. Return numpy arrays.
    If select_features is specified, a subset of all available
    features are returned:
        select_features = 'basic': Return 4-vectors, MET, phistar
        select_features = 'extended': Also add masses
    """
    
    
    X = np.array([])
    Y = np.array([])
    features = None

    print 'Opening files:'
    for fin in files:
        hf = h5py.File(fin, 'r')
        data = hf.get('data')
        features = data.attrs['feature_names']
        
        data = np.array(data)
        
        # Add up the data
        if len(X):
            assert data.shape[1] == X.shape[1]
            X = np.vstack((X, data))
        else:
            X = data
         
        print ' %s (%d events)' % (fin, data.shape[0])

    # Extract targets
    Y = np.array(X[:, -1])
    X = np.delete(X, -1, axis=1) # remove targets from data
    
    X, features = select_features(X, features, include_mass_variables=include_mass)
    
    print ' Got %d events.' % X.shape[0]

    return X, Y, features




