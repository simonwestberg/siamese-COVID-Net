import numpy as np

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
