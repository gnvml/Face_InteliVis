'''
This function is to add new person to exis csv file (if not then create new)

Structure: 
    Id, Image, Label

'''

import pandas as pd
import os

DATA_PATH = 'DataFace'
CSV_PATH = 'CSV_LIST'

def toCsvFile():

    if not os.path.isdir(CSV_PATH):
        os.makedirs(CSV_PATH)

    Names = []
    Images = []
    ID = []
    count = 0

    for name in os.listdir(DATA_PATH):
        for image in os.listdir(os.path.join(DATA_PATH, name)):
            Names.append(name)
            Images.append(image)
            ID.append(count)
        count += 1
    df = pd.DataFrame({'Name':Names,'Image':Images, 'ID':ID})
    df.to_csv(CSV_PATH + '/CSV_List.csv',index=False)
    print(df)


# toCsvFile()