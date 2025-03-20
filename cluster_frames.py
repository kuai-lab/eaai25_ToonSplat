import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imread
import pdb

import torch
from torchvision.io import read_image
from torchvision.transforms.functional import resize, to_tensor, normalize
from torchvision.models import resnet101
from tqdm import tqdm


def image_clustering_from_folder(folder_path, n_clusters):

    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png')]
    images = [imread(path) for path in image_paths]

    X = np.array([image.flatten() for image in images])


    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    labels = kmeans.labels_
    for idx, value in enumerate(labels):
        print(f"Index: {idx}, Value: {value}")
    # pdb.set_trace()

    
def calculate_image_feature_distribution(image_folder, image_encoder_model, save_path):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
    features = []
    for image_file in tqdm(image_files):
        # pdb.set_trace()
        image_path = os.path.join(image_folder, image_file)
        image = read_image(image_path)
        image = resize(image, (224, 224))
        image = normalize(image.type(torch.float32)/255.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = image.unsqueeze(0)
        with torch.no_grad():
            feature = image_encoder_model(image).squeeze().numpy()
        features.append(feature)


    features = np.array(features)
    mean = np.mean(features, axis=1)
    variance = np.var(features, axis=1)
    
    mean_all = np.mean(mean)
    variance_all = np.var(variance)    
    top_k_indices = np.argsort(np.abs(mean-mean_all))[-5:]

folder_path = './data/ToonVid/AladdinDance/images'  # masked_images_ours
image_encoder = resnet101(pretrained=True)
image_encoder.eval()
save_path = 'feature.png'


n_clusters = 3
calculate_image_feature_distribution(folder_path, image_encoder, save_path)