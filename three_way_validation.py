import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras

def get_validation_pairs(trials, X_covid_val, X_normal_val, X_pneumonia_val):
    """ 
    Returns trials amount of validation triplet-pairs for 3-way one shot learning.
    If trials=3, the function returns 
    X_val = [   [[cov_img, cov_img], [cov_img, norm_img], [cov_img, pnem_img]], 
                [[norm_img, cov_img], [norm_img, norm_img], [norm_img, pnem_img]], 
                [[pnem_img, cov_img], [pnem_img, norm_img], [pnem_img, pnem_img]]] 
    Y_val = [   [0, 1, 1], 
                [1, 0, 1], 
                [1, 1, 0]]
    i.e., we get three trials of 3-way one shot learning validation tasks.  
    """

    N_covid_val, h, w = X_covid_val.shape
    N_normal_val = X_normal_val.shape[0]
    N_pneumonia_val = X_pneumonia_val.shape[0]

    if trials % 3 != 0:
        raise ValueError("Trials not a multiple of 3.")

    if trials // 3 >= N_covid_val:
        raise ValueError("Not enough covid images.")

    indices_covid = np.arange(N_covid_val)
    indices_normal = np.arange(N_normal_val)
    indices_pneumonia = np.arange(N_pneumonia_val)

    X_val = []
    Y_val = []

    for i in range(trials // 3):
        # COVID
        # First extract a covid-image to predict the class of
        index = np.random.choice(indices_covid)
        covid_img = X_covid_val[index, :, :].reshape(h, w, 1)
        indices_covid = indices_covid[indices_covid != index]

        # Extract a covid, normal, and pneumonia image to compare against in 3-way learning
        index_covid = np.random.choice(indices_covid)
        img1 = X_covid_val[index_covid, :, :].reshape(h, w, 1)

        index_normal = np.random.choice(indices_normal)
        img2 = X_normal_val[index_normal, :, :].reshape(h, w, 1)

        index_pneumonia = np.random.choice(indices_pneumonia)
        img3 = X_pneumonia_val[index_pneumonia, :, :].reshape(h, w, 1)
        
        # Append the triplet of image-pairs to X_val, and corresponding label to Y_val
        X_val.append([[covid_img, img1], [covid_img, img2], [covid_img, img3]]) 
        Y_val.append([0, 1, 1])
        
        # NORMAL
        # First extract a normal-image to predict the class of
        index = np.random.choice(indices_normal)
        normal_img = X_normal_val[index, :, :].reshape(h, w, 1)
        indices_normal = indices_normal[indices_normal != index]

        # Extract a covid, normal, and pneumonia image to compare against in 3-way learning
        index_covid = np.random.choice(indices_covid)
        img1 = X_covid_val[index_covid, :, :].reshape(h, w, 1)

        index_normal = np.random.choice(indices_normal)
        img2 = X_normal_val[index_normal, :, :].reshape(h, w, 1)

        index_pneumonia = np.random.choice(indices_pneumonia)
        img3 = X_pneumonia_val[index_pneumonia, :, :].reshape(h, w, 1)

        X_val.append([[normal_img, img1], [normal_img, img2], [normal_img, img3]])
        Y_val.append([1, 0, 1])
        
        # PNEUMONIA
        # First extract a pneumonia-image to predict the class of
        index = np.random.choice(indices_pneumonia)
        pneumonia_img = X_pneumonia_val[index, :, :].reshape(h, w, 1)
        indices_pneumonia = indices_pneumonia[indices_pneumonia != index]

        # Extract a covid, normal, and pneumonia image to compare against in 3-way learning
        index_covid = np.random.choice(indices_covid)
        img1 = X_covid_val[index_covid, :, :].reshape(h, w, 1)

        index_normal = np.random.choice(indices_normal)
        img2 = X_normal_val[index_normal, :, :].reshape(h, w, 1)

        index_pneumonia = np.random.choice(indices_pneumonia)
        img3 = X_pneumonia_val[index_pneumonia, :, :].reshape(h, w, 1)

        X_val.append([[pneumonia_img, img1], [pneumonia_img, img2], [pneumonia_img, img3]])
        Y_val.append([1, 1, 0])
    
    return X_val, Y_val

# Validation callback
class Validation(keras.callbacks.Callback):
    """A custom validation callback for 3-way one-shot validation used in the 
    training of a siamese network."""
    
    def __init__(self, validation_data, tr_batches_per_epoch):
        super(keras.callbacks.Callback, self).__init__()
        self.validation_data = validation_data
        self.X_val = self.validation_data[0]
        self.Y_val = self.validation_data[1]
        self.trials = len(self.Y_val)
        self.val_accuracy = []
        self.tr_batches_per_epoch = tr_batches_per_epoch
        
        self.max_val_acc = 0

        self.confusion_matrices = []
        self.confusion_matrices_perc = []

    def on_epoch_end(self, epoch, logs=None):
        self.validation()
        accuracy = self.val_accuracy[-1]
        print("\n3-way validation accuracy {} trials - epoch: {} - accuracy: {}".format(self.trials, epoch, accuracy))
        print(self.confusion_matrices_perc[-1])
        
        # Save model if accuracy is larger than previous max validation accuracy
        #if accuracy > self.max_val_acc:
        #    self.max_val_acc = accuracy
        #    self.model.save("Models/siameseNet")

    def validation(self):
        correct = 0
        true_labels = []
        predicted_labels = []
        for i, triplet in enumerate(self.X_val):
          
            pair1 = triplet[0]
            pair2 = triplet[1]
            pair3 = triplet[2]

            sim1 = self.model.predict(x=[[pair1[0]], [pair1[1]]])
            sim2 = self.model.predict(x=[[pair2[0]], [pair2[1]]])
            sim3 = self.model.predict(x=[[pair3[0]], [pair3[1]]])
            
            predict_triplet = [sim1, sim2, sim3]
            Y_triplet = self.Y_val[i]
            
            prediction = np.argmin(predict_triplet)
            truth = np.argmin(Y_triplet)
            
            true_labels.append(truth)
            predicted_labels.append(prediction)

            if prediction == truth:
                correct += 1
        
        accuracy = correct / self.trials
        conf = confusion_matrix(true_labels, predicted_labels)
        self.val_accuracy.append(np.round(accuracy, 3))
        self.confusion_matrices.append(conf)
        self.confusion_matrices_perc.append(np.round(conf / np.sum(conf, axis=1), 2))
