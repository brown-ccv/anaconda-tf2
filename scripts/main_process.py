#!/usr/bin/python

import sys
import importlib
from datagenerator import datagenerator
import output
import training

#import cnnmodel implementation 
cnnmodel = importlib.import_module(sys.argv[1])

#create model
model = cnnmodel.cnnmodel_impl()
#build network
model.buildNetwork()

#load and create imagegenerators
data  = datagenerator(model)

#Train the network
history = training.train(model,data)

#output
output.save(model)
output.plot(history, model)
output.predict(model, data)

#delete the imagegenerators
del data
#delete the model
del model