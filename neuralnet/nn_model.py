# The neural network model to use

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.layers import AlphaDropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU


def create_model(xdim, ydim):
    
    model = Sequential()
    
    initializer = 'he_uniform'
    drop = 0.375
    
    output_activation = 'softmax'
    lossfn = 'categorical_crossentropy'
    metric = ['categorical_accuracy']
    
    model.add(Dense(500, input_dim=xdim, kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(drop))
    
    # Hidden layer 1
    model.add(Dense(1000, kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(drop))
    
    # Hidden layer 2
    model.add(Dense(100, kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(drop))
    
    # Output layer
    model.add(Dense(ydim, kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(Activation(output_activation))
    
    optim = Adam(lr=5e-5)

    model.compile(optimizer=optim, loss=lossfn, metrics=metric)
    
    return model
