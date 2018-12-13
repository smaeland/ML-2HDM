# pylint: disable=C0303

import os
import numpy as np
from argparse import ArgumentParser
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from neuralnet.utilities import get_dataset_from_path
from neuralnet.nn_model import create_model

try:
    import matplotlib.pyplot as plt
    disable_graphics = False
except ImportError:
    disable_graphics = True



def train(traindir, nametag, scalerfile, recreate_scaler=False, verbose=0):
    """
    Train a network.
    
    Arguments:
        traindir: Path to training data
        nametag: Name used for output file
        scalerfile: Name of file to read/write scaler instance
        recreate_scaler: Re-fit the scaler
        verbose: See Sequential.fit()
    """
    
    # Get data
    X, Y, _ = get_dataset_from_path(traindir,
                                    pattern='model_*_merged',
                                    include_mass=False)

    
    # Convert targets to binary class matrix
    Y = to_categorical(Y, num_classes=2)

    
    # Split train/test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.50)

    # Preprocess
    # First get the scaler object, recreate if it doesn't exist or if required
    if not os.path.exists(scalerfile) or recreate_scaler:
        scaler = preprocessing.StandardScaler().fit(X_train)
        with open(scalerfile, 'wb') as hfile:
            pickle.dump(scaler, hfile)
    else:
        with open(scalerfile) as hfile:
            scaler = pickle.load(hfile)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    # Create network model
    model = create_model(X_train.shape[1], Y_train.shape[1])
    
    # Callbacks
    reduce_lr = ReduceLROnPlateau(min_lr=1e-7, verbose=1, cooldown=10)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
    
    # Train
    loss_history = model.fit(X_train, Y_train,
                             epochs=100,
                             batch_size=264,
                             validation_data=(X_test, Y_test),
                             callbacks=[reduce_lr, early_stop],
                             verbose=verbose)
    
    # Store training history
    with open('loss_history_%s.pkl' % nametag, 'wb') as hfile:
        pickle.dump(loss_history.history, hfile)
    print 'Training history written to loss_history_%s.pkl' % nametag

    # Save model
    if not '.h5' in nametag:
        nametag += '.h5'
    model.save('keras_model_%s' % nametag)

    # Compute ROC AUC
    roc_auc = roc_auc_score(Y_test, model.predict(X_test))
    print 'Model AUC =', roc_auc
    
    if not disable_graphics:
        fpr, tpr, _ = roc_curve(Y_test, model.predict(X_test))
        roc_auc = auc(fpr, tpr)    

        fig = plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        fig.show()
        #plt.savefig('roc_%s.pdf' % name)
    


if __name__ == '__main__':
    
    parser = ArgumentParser(description='Train networks to separate signal')
    parser.add_argument('-i', '--input', help='Path to directory containing training set', default='../../generate_samples/450GeV/train')
    parser.add_argument('-gpu', '--gpu', help='Enable GPU', action='store_true')
    parser.add_argument('-os', '--overwrite_scaler', help='Recreate scaler instance', action='store_true')
    parser.add_argument('-v', '--verbose', help='Verbose output', type=int, default=0)
    pargs = parser.parse_args()
    
    # Enable GPU
    if pargs.gpu:
        import tensorflow as tf
        gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.4)
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    
    # Start training
    train(traindir=pargs.input,
          nametag='100ep.h5',
          scalerfile='scaler_common.pkl',
          recreate_scaler=pargs.overwrite_scaler,
          verbose=pargs.verbose)
    
    print 'Done.'
    
    
    
