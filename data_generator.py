import tensorflow as tf
from tensorflow import keras
import numpy as np

class DataGenerator(keras.utils.Sequence):
    """Data generator for a siamese network architecture. """
    
    def __init__(self, X_normal, X_pneumonia, X_covid, num_channels=1, batch_size=32, samples_per_epoch=5000):
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        
        self.X_normal = X_normal
        self.X_pneumonia = X_pneumonia
        self.X_covid = X_covid

        self.h = self.X_normal.shape[1]
        self.w = self.X_normal.shape[2]
        
        self.N_normal = self.X_normal.shape[0]
        self.N_pneumonia = self.X_pneumonia.shape[0]
        self.N_covid = self.X_covid.shape[0]

        self.same_combinations = self.get_same_combinations(self.N_covid)
        self.different_combinations = self.get_different_combinations(self.N_covid)

        self.epoch_covid = self.X_covid.reshape(-1, self.h, self.w, num_channels)

        # Extract N_covid amount of normal and pneumonia imgs randomly
        idx_normal = np.random.choice(self.N_normal, size=self.N_covid, replace=False)
        idx_pneumonia = np.random.choice(self.N_pneumonia, size=self.N_covid, replace=False)
        self.epoch_normal = self.X_normal[idx_normal, :, :].reshape(-1, self.h, self.w, self.num_channels)
        self.epoch_pneumonia = self.X_pneumonia[idx_pneumonia, :, :].reshape(-1, self.h, self.w, self.num_channels)

        # Shuffle combinations
        np.random.shuffle(self.same_combinations)
        np.random.shuffle(self.different_combinations)

    def get_same_combinations(self, num):
        lis = []
        for i in range(num):
            for j in range(i+1, num):
                lis.append([i, j])
        return lis

    def get_different_combinations(self, num):
        lis = []
        for i in range(num):
            for j in range(num):
                lis.append([i, j])
        return lis

    def __len__(self):
        """Return number of batches per epoch."""
        return self.samples_per_epoch // self.batch_size
    
    def __getitem__(self, index):
        """Return a batch with corresponding index.
        First batch has index 0, second has index 1, and so on.
        """
        pairs_a = []
        pairs_b = []
        labels = []

        n_same = self.batch_size // 2
        n_different = self.batch_size // 2

        # Generate same-class pairs
        same = self.same_combinations[n_same*index:n_same*(index+1)]

        for i, pair in enumerate(same):
            if i < n_same // 3:
                # Covid
                pairs_a.append(self.epoch_covid[pair[0]])
                pairs_b.append(self.epoch_covid[pair[1]])
            elif i < 2 * (n_same // 3):
                # Normal
                pairs_a.append(self.epoch_normal[pair[0]])
                pairs_b.append(self.epoch_normal[pair[1]])
            elif i < 3 * (n_same // 3):
                # Pneumonia
                pairs_a.append(self.epoch_pneumonia[pair[0]])
                pairs_b.append(self.epoch_pneumonia[pair[1]])
            else:
                # Choose random class for remaining pairs
                k = np.random.randint(3)
                if k == 0:
                    pairs_a.append(self.epoch_covid[pair[0]])
                    pairs_b.append(self.epoch_covid[pair[1]])
                elif k == 1:
                    pairs_a.append(self.epoch_normal[pair[0]])
                    pairs_b.append(self.epoch_normal[pair[1]])
                else:
                    pairs_a.append(self.epoch_pneumonia[pair[0]])
                    pairs_b.append(self.epoch_pneumonia[pair[1]])
            labels.append(0)
        
        # Generate different-class pairs
        different = self.different_combinations[n_different*index:n_different*(index+1)]

        for i, pair in enumerate(different):
            if i < n_different // 3:
                # Covid-normal
                pairs_a.append(self.epoch_covid[pair[0]])
                pairs_b.append(self.epoch_normal[pair[1]])
            elif i < 2 * (n_different // 3):
                # Covid-pneumonia
                pairs_a.append(self.epoch_covid[pair[0]])
                pairs_b.append(self.epoch_pneumonia[pair[1]])
            elif i < 3 * (n_different // 3):
                # Normal-pneumonia
                pairs_a.append(self.epoch_normal[pair[0]])
                pairs_b.append(self.epoch_pneumonia[pair[1]])
            else:
                k = np.random.randint(3)
                if k == 0:
                    # Covid-normal
                    pairs_a.append(self.epoch_covid[pair[0]])
                    pairs_b.append(self.epoch_normal[pair[1]])
                elif k == 1:
                    # Covid-pneumonia
                    pairs_a.append(self.epoch_covid[pair[0]])
                    pairs_b.append(self.epoch_pneumonia[pair[1]])
                else:
                    # Normal-pneumonia
                    pairs_a.append(self.epoch_normal[pair[0]])
                    pairs_b.append(self.epoch_pneumonia[pair[1]])
            labels.append(1)

        pairs_a = np.asarray(pairs_a)
        pairs_b = np.asarray(pairs_b)
        labels = np.asarray(labels)
    
        # Shuffle the data
        permutation = np.random.permutation(len(labels))
        pairs_a = pairs_a[permutation, :, :, :]
        pairs_b = pairs_b[permutation, :, :, :]
        labels = labels[permutation]

        return [pairs_a, pairs_b], labels
  
    def on_epoch_end(self):
        # Extract N_covid amount of normal and pneumonia imgs randomly
        idx_normal = np.random.choice(self.N_normal, size=self.N_covid, replace=False)
        idx_pneumonia = np.random.choice(self.N_pneumonia, size=self.N_covid, replace=False)
        self.epoch_normal = self.X_normal[idx_normal, :, :].reshape(-1, self.h, self.w, self.num_channels)
        self.epoch_pneumonia = self.X_pneumonia[idx_pneumonia, :, :].reshape(-1, self.h, self.w, self.num_channels)

        # Shuffle combinations
        np.random.shuffle(self.same_combinations)
        np.random.shuffle(self.different_combinations)
