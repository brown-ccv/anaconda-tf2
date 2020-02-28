from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot(history, model):
    with PdfPages(model.output +'plot.pdf') as pdf:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        pdf.savefig();
        plt.close()

def save(model):
    model.model.save_weights(model.output +'weights.h5')

        
import numpy as np
def predict(model,data):
    #predict test_data
    def probas_to_classes(y_pred):
        return np.array([1 if p > 0.5 else 0 for p in y_pred])

    # run testdata
    prediction = model.model.predict(data.testgenerator)
    pred_classes = probas_to_classes(prediction)

    print(data.testgenerator.class_indices)

    for i in range(0,len(prediction)):
        
        pred_class = list(data.testgenerator.class_indices.keys())[list(data.testgenerator.class_indices.values()).index(pred_classes[i])]
        
        print(data.test_filenames[i] + ' - ' + data.test_labels[i] + ' - ' + str(prediction[i])+ ' - pred: ' + pred_class)