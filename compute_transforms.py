import scipy.io as sio
import numpy as np

# data = sio.loadmat('features.mat')
data = sio.loadmat('surf_features.mat')
print(data.keys())
features = data['features']
print(features.shape)

