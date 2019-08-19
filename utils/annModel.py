'''
Architecture Ann model for face recognition

'''
import os
import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras.layers import Dense, Dropout
from keras.models import Sequential, model_from_json

BATCH_SIZE = 32
EPOCHS = 100
INPUT_DIM = 128


def save_model(model, cross):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def Training(x_train, y_train, df, NUMBER_OF_CLASSES):
    '''
    Training ANN model and automatically save model in cache foler
    '''

    
    model = Sequential()
    model.add(Dense(2048, input_dim=INPUT_DIM, init='uniform', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=NUMBER_OF_CLASSES, init='uniform', activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS)
    
    save_model(model, "{batch}_{epoch}_{suffix}".format(batch=BATCH_SIZE, epoch=EPOCHS, suffix='none'))



