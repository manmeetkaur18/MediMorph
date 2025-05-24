from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import os
import glob

def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    expression_map = {
        'AN': 'anger',
        'DI': 'disgust',
        'FE': 'fear',
        'HA': 'happiness',
        'NE': 'neutral',
        'SA': 'sadness',
        'SU': 'surprise'
    }

    for filename in glob.glob(os.path.join(data_dir, '*.tiff')):
        try:
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48)) # Resize for consistency
            img = img / 255.0  # Normalize pixel values
            images.append(img)

            # Extract label from filename (e.g., 'KL.AN1.39.tiff' -> 'AN')
            parts = filename.split('.')
            if len(parts) >= 2:
                expression_code = parts[1][:2]  # Take the first two characters after the first dot
                if expression_code in expression_map:
                    labels.append(expression_map[expression_code])
                else:
                    print(f"Warning: Unknown expression code '{expression_code}' in {filename}")
            else:
                print(f"Warning: Could not extract expression code from {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    images = np.array(images).reshape(-1, 48, 48, 1) # Reshape for CNN (height, width, channels)
    labels = np.array(labels)

    # Encode labels to numerical values
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(labels)
    categorical_labels = to_categorical(integer_labels, num_classes=len(expression_map))

    return images, categorical_labels, label_encoder.classes_

data_dir='jaffe'

images, labels, class_names = load_and_preprocess_data(data_dir)


loaded_model = load_model('C:\\Users\\DELL\\OneDrive\\Desktop\\medical_snapchat\\models\\jaffe_expression_model.h5')
print("Trained model loaded successfully.")

def preprocess_new_image(image_path, target_size=(48, 48)):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return None
        img_resized = cv2.resize(img, target_size)
        img_normalized = img_resized / 255.0
        img_reshaped = img_normalized.reshape(1, target_size[0], target_size[1], 1) # (batch_size, height, width, channels)
        return img_reshaped
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

# Example usage:
new_image_path = 'C:\\Users\\DELL\\OneDrive\\Desktop\\medical_snapchat\\models\\WIN_20250426_09_32_26_Pro.jpg'  # Replace with the path to your new image
processed_image = preprocess_new_image(new_image_path)

if processed_image is not None:
    print("New image preprocessed successfully.")
else:
    print("Failed to preprocess the new image.")


if processed_image is not None:
    predictions = loaded_model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_expression = class_names[predicted_class_index]
    probability = predictions[0][predicted_class_index]

    print(f"Predicted Expression: {predicted_expression}")
    print(f"Probability: {probability:.4f}")

    # You can also visualize the probability distribution across all classes
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, predictions[0])
    plt.xlabel('Facial Expression')
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.show()