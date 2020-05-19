import random
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os, sys
def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

class datagenerator:    
    def __init__(self, model):

        # Import Data from the csv
        train_filenames = []
        self.test_filenames = []
        train_labels = []
        self.test_labels = []

        #read csv file
        with open(model.path + 'descriptor.txt') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                #skip CNT for now
                dataset_folder = splitall(row[2])
                if row[1] in  model.conditions:
                    if dataset_folder[len(dataset_folder) - 2] in model.trainingsets:
                        train_filenames.append(model.path + model.subfolder + model.channel+ row[0] + model.ending) 
                        train_labels.append(row[1])
                    elif dataset_folder[len(dataset_folder) - 2] in model.testsets:
                        self.test_filenames.append(model.path + model.subfolder + model.channel+ row[0] + model.ending) 
                        self.test_labels.append(row[1])

        #split in train and test data
        if not model.testsets:
            train_filenames, self.test_filenames, train_labels, self.test_labels = train_test_split(train_filenames, train_labels,train_size = 1.0 - model.testsize,random_state=42)

        #create panda dataFrame for traindata
        d_train= {'filename': train_filenames, 'class': train_labels}
        df_train = pd.DataFrame(data=d_train)
        print(df_train.groupby(['class']).size())
        print(df_train)
        
        #create panda dataFrame for testdata
        d_test = {'filename': self.test_filenames, 'class': self.test_labels}
        df_test = pd.DataFrame(data=d_test)
        print(df_test.groupby(['class']).size())
        print(df_test)
        
        #create Image generator
        datagen = ImageDataGenerator(
            rescale=model.rescale,  # set values to 0-1
            rotation_range=model.rotation_range, # rotation does not matter
            width_shift_range=model.width_shift_range, # 20 % border
            height_shift_range=model.height_shift_range,
            horizontal_flip=model.horizontal_flip, # flip does not matter
            vertical_flip=model.vertical_flip, # flip does not matter
            fill_mode="nearest", 
            validation_split=model.validationsplit) 
        
        #create random seed
        seed = random.seed()

        #Note. not sure if these have to be reshuffled during iterations
        #create generator for training data
        self.traingenerator = datagen.flow_from_dataframe(
                dataframe=df_train,
                x_col="filename",
                y_col="class",
                target_size=(model.imagesize, model.imagesize),
                batch_size=model.batchsize,
                class_mode='binary',
                subset='training',
                seed = seed,
                shuffle=True,
                color_mode = 'grayscale' if model.nbchannels == 1 else 'rgb' )

        #create generator for validation data
        self.validationgenerator = datagen.flow_from_dataframe(
                dataframe=df_train,
                x_col="filename",
                y_col="class",
                target_size=(model.imagesize, model.imagesize),
                batch_size=model.batchsize,
                class_mode='binary',
                subset='validation',
                seed = seed,
                shuffle=True,
                color_mode = 'grayscale' if model.nbchannels == 1 else 'rgb' )


        #create Image generator
        datagen_test = ImageDataGenerator(
                rescale=model.rescale)  # set values to 0-1

        #Note. not sure if these have to be reshuffled during iterations
        #create generator for training data
        self.testgenerator = datagen_test.flow_from_dataframe(
                dataframe=df_test,
                x_col="filename",
                y_col="class",
                target_size=(model.imagesize, model.imagesize),
                shuffle = False,
                color_mode = 'grayscale' if model.nbchannels == 1 else 'rgb' )
