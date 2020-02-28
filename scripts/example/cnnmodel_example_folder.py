import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Dropout, Flatten, add
from tensorflow.keras.models import Model

class cnnmodel_impl():
    def __init__(self):
            
        #INPUT
        #path to the main data folder
        self.path = 'D:\\work\\48h_output\\'
        #subfolder for the data, e.g. 'max/' or 'mean/'
        self.subfolder = 'max/'
        #folder for the channel eg, "ch3/" or empty '' for rgb
        self.channel = '' 
        #ending of the images
        self.ending = '.png'
        #folder where output is saved, use './' for current directory
        self.output = '.\\'
        #Number of channels the image has 3 for rgb processing, 1 for grayscale 
        self.nbchannels = 3

        #TEST - TRAIN - VALIDATIONSPLIT
        #Percentage of images for a train set, leftover is used as test dataset
        self.trainsize = 0.9; 
        #Percentage of images for a validation set, leftover is used as training dataset
        self.validationsplit = 0.3
        
        #IMAGEGENERATOR - should not need adjustement but might be nice for other projects
        #scaling should be set to bring values in the range of 0-1
        self.rescale=1./255  # set values to 0-1
        #Maximum rotation of images
        self.rotation_range=180 
        #Maximum translation of the image along x-axis
        self.width_shift_range=0.2 
        #Maximum translation of the image along y-axis
        self.height_shift_range=0.2
        #Can the image flip horizontally
        self.horizontal_flip=True
        #Can the image flip vertically
        self.vertical_flip=True
        
        #HYPERPARAMETERS
        #Size of the images which are being processed
        self.imagesize = 540
        #Batch size
        self.batchsize = 16
        #Number of epochs
        self.nbepochs = 1
        #Number of epochs
        self.learningrate = 0.0001
        
    def buildNetwork(self):
        #CREATE NETWORK   
        STRIDE = 1
        CHANNEL_AXIS = 3

        def res_layer(x ,filters,pooling = False,dropout = 0.0):
            temp = x
            temp = Conv2D(filters,(5,5),strides = STRIDE,padding = "same")(temp)
            temp = BatchNormalization(axis = CHANNEL_AXIS)(temp)
            temp = Activation("relu")(temp)
            temp = Conv2D(filters,(5,5),strides = STRIDE,padding = "same")(temp)

            x = add([temp,Conv2D(filters,(5,5),strides = STRIDE,padding = "same")(x)])
            if pooling:
                x = MaxPooling2D((4,4))(x)
            if dropout != 0.0:
                x = Dropout(dropout)(x)
            x = BatchNormalization(axis = CHANNEL_AXIS)(x)
            x = Activation("relu")(x)
            return x

        inp = Input(shape = (self.imagesize,self.imagesize, self.nbchannels))
        x = inp
        x = Conv2D(16,(3,3),strides = STRIDE,padding = "same")(x)
        x = BatchNormalization(axis = CHANNEL_AXIS)(x)
        x = Activation("relu")(x)
        x = res_layer(x,8,dropout = 0.4,pooling = True)
        x = res_layer(x,16,dropout = 0.4,pooling = True)
        x = res_layer(x,32,dropout = 0.4,pooling = True)
        x = Flatten()(x)
        x = Dropout(0.4)(x)
        x = Dense(2048,activation = "relu", bias_initializer='zeros')(x)
        x = Dropout(0.4)(x)
        x = Dense(1,activation = "sigmoid")(x)

        self.model = Model(inp,x,name = "Resnet")

        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learningrate), 
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        self.model.summary()