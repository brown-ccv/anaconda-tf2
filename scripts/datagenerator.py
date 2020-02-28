import random
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class datagenerator:    
    def __init__(self, model):
        
        # Import Data from the csv
        labels = []
        images = []

        #read csv file
        with open(model.path + 'descriptor.txt') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                #skip CNT for now
                if row[1] != 'CNT':
                    images.append(model.path + model.subfolder + model.channel+ row[0] + model.ending) 
                    labels.append(row[1])

        #split in train and test data
        train_filenames, self.test_filenames, train_labels, self.test_labels = train_test_split(images, labels,train_size = model.trainsize,random_state=42)

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
