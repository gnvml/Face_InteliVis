'''
This function is to merge all embeded face in 1 file for training data

'''

import pandas as pd
import numpy as np
import os
from multiprocessing import Pool, cpu_count

CSV_PATH = 'CSV_LIST/CSV_List.csv'
EMB = 'Embeded_Face'

def embededAll():

    train_data = []
    
    for name in os.listdir(EMB):    
        for embFile in os.listdir(os.path.join(EMB, name)):
            full_file_path = os.path.join(EMB, name, embFile)
            emb = np.load(full_file_path)
            train_data.append(emb)

    train_data = np.array(train_data)
    print(train_data.shape)
    np.save('train_data.npy', train_data)

embededAll()