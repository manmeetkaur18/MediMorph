import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib
import json
import os

# Define paths (should be the same as in the training script)
feature_extraction_model_path = 'C:\\Users\\DELL\\OneDrive\\Desktop\\medical_snapchat\\models\\vgg16_feature_extractor.h5'
kmeans_model_path = 'C:\\Users\\DELL\\OneDrive\\Desktop\\medical_snapchat\\models\\skin_condition_kmeans.joblib'
cluster_mapping_path = 'C:\\Users\\DELL\\OneDrive\\Desktop\\medical_snapchat\\models\\cluster_label_mapping.json'
img_height = 128
img_width = 128

# Load the feature extraction model
try:
    feature_extractor = tf.keras.models.load_model(feature_extraction_model_path)
    print(f"Feature extraction model loaded from: {feature_extraction_model_path}")
except Exception as e:
    print(f"Error loading feature extraction model: {e}")
    feature_extractor = None

# Load the K-Means model
try:
    kmeans_model = joblib.load(kmeans_model_path)
    print(f"K-Means model loaded from: {kmeans_model_path}")
except Exception as e:
    print(f"Error loading K-Means model: {e}")
    kmeans_model = None

# Load the cluster label mapping
try:
    with open(cluster_mapping_path, 'r') as f:
        cluster_label_mapping = json.load(f)
    print(f"Cluster label mapping loaded from: {cluster_mapping_path}")
except FileNotFoundError:
    print(f"Error: Cluster label mapping file not found at {cluster_mapping_path}")
    cluster_label_mapping = None
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {cluster_mapping_path}")
    cluster_label_mapping = None

def extract_features(model, img_path):
    try:
        img = image.load_img(img_path, target_size=(img_height, img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        features = model.predict(img_array)
        return features.flatten()
    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}")
        return None

def predict_skin_condition(img_path):
    if feature_extractor is None or kmeans_model is None or cluster_label_mapping is None:
        print("Error: One or more models/mappings not loaded. Cannot predict.")
        return None

    features = extract_features(feature_extractor, img_path)
    if features is not None:
        try:
            predicted_cluster = kmeans_model.predict(features.reshape(1, -1))[0]
            return cluster_label_mapping.get(predicted_cluster, 'unknown')
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    else:
        return None

# Example usage in the new file:
new_image_path = 'C:\\Users\\DELL\\OneDrive\\Desktop\\medical_snapchat\\models\\ds00190_-ds00439_im01723_r7_skincthu_jpg.jpg'  
try:
    prediction = predict_skin_condition(new_image_path)
    if prediction:
        print(f"Predicted condition for {new_image_path}: {prediction}")
except Exception as e:
    print(f"An error occurred: {e}")