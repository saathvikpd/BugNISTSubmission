import os

import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.cluster import DBSCAN
import pickle
import segmentation_models_3D as sm
from patchify import patchify, unpatchify
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import os
import tensorflow as tf
import keras

def predict(data_path, pkl_path):
    """Predict bugnist data.

    Parameters
    ----------
    data_path : str
        Path to data directory,e.g.
        "bugnist2024fgvc/BugNIST_DATA/validation" or
        "bugnist2024fgvc/BugNIST_DATA/test"
    pkl_path : str
        Path to additional required data. Here, it's
        the weights of the torch model, "model.pkl"

    Returns
    -------
    df_pred : pd.DataFrame
        Prediction formated as required by the score function.
    """

    # Load model
    with open(pkl_path, 'rb') as f:
        models = pickle.load(f)

    # Labels - has to match what was used
    #   during training, be careful with this
    label_map = \
    {1: 'SL',
     2: 'PP',
     3: 'BP',
     4: 'GH',
     5: 'BL',
     6: 'AC',
     7: 'MA',
     8: 'BC',
     9: 'CF',
     10: 'ML',
     11: 'BF',
     12: 'WO'}

    # Paths to mixtures
    files = os.listdir(data_path)
    files.sort()
    files = [i for i in files if i.endswith(".tif")]
    files = files[:2] # first two files as an example

    # Predict bugs in each file
    rows = []
    for file in files:
        preds = []
        for mi, m in enumerate(models):
            stage = make_pred(m, f"{data_path}/{file}")
            if np.isnan(stage).ravel().mean() == 0:
                preds += [stage]
        all_preds = np.mean(np.array(preds), axis = 0)
        reconstructed_image = np.argmax(all_preds, axis = 3)

        classes, class_counts = np.unique(reconstructed_image, return_counts = True)
        class_centers = []
        for c, c_count in zip(classes, class_counts):
            if c != 0:
                data = np.array(np.where(reconstructed_image == c)).T
                db = DBSCAN(eps = 0.8, min_samples = 5)
                preds = db.fit_predict(data)
                if np.unique(preds).shape[0] <= 1:
                    center = data.mean(axis = 0).round(2)
                    class_centers += [[label_map[c]] + list(center)]
                else:
                    unq_bugs = np.unique(preds)
                    for u in unq_bugs:
                        sub_data = data[preds == u]
                        center = sub_data.mean(axis = 0).round(2)
                        class_centers += [[label_map[c]] + list(center)]

        class_centers = sorted(class_centers)
        res = ""
        for c in class_centers:
            res += c[0].lower() + ";"
            res += str(c[1]) + ";"
            res += str(c[2]) + ";"
            res += str(c[3]) + ";"

        res = res[:-1]
        
        rows += [res]


    # This must be compatible with the provided
    #   scoring script on kaggle
    df_pred = pd.DataFrame({
        "file_name": files[:2],
        "centerpoints": rows
    })

    return df_pred

def make_pred(model, path, patch_size = 32):
    BACKBONE = "resnet18"
    preprocess_input = sm.get_preprocessing(BACKBONE)
    large_image = imread(path)
    print(large_image.shape)
    large_image = np.pad(large_image, pad_width = ((0, 0), (2, 2), (2, 2)))
    patches = patchify(large_image, (patch_size, patch_size, patch_size), step=patch_size)  
    predicted_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            for k in range(patches.shape[2]):
                single_patch = patches[i,j,k, :,:,:]
                single_patch_3ch_input = preprocess_input(np.expand_dims(single_patch, axis=0))
                single_patch_prediction = model.predict(single_patch_3ch_input, verbose=0)
                predicted_patches.append(single_patch_prediction[0])
    predicted_patches = np.array(predicted_patches)
    predicted_patches_reshaped = np.reshape(predicted_patches,
                                            (patches.shape[0], patches.shape[1], patches.shape[2],
                                             patches.shape[3], patches.shape[4], patches.shape[5],
                                             13
                                            ))
    rec_imgs = []
    for i in range(13):
        rec_imgs += [unpatchify(predicted_patches_reshaped[:, :, :, :, :, :, i], large_image.shape)]
    reconstructed_image = np.stack(rec_imgs, axis = 3)
    return reconstructed_image

# # Model to predict bugs. This must be accessible
# #   from main (not inside the predict function)
# #    and  is needed to unpack the pkl file
# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.seq = nn.Sequential(
#             nn.Conv3d(1, 3, kernel_size=5, stride=1),
#             nn.MaxPool3d(2),
#             nn.Conv3d(3, 3, kernel_size=4, stride=1),
#             nn.MaxPool3d(2),
#             nn.Conv3d(3, 3, kernel_size=3, stride=1),
#             nn.MaxPool3d(4),
#             nn.Flatten(),
#             nn.Linear(375, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 12)
#         )
#     def forward(self, x):
#         return self.seq(x)

# data_path = "./validation/"
# pkl_path = "model.pkl"

# df_pred = predict(data_path, pkl_path)