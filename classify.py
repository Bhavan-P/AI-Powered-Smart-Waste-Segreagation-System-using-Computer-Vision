import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define waste categories (Ensure your dataset follows this structure)
categories = ["Plastic", "paper", "metal", "glass"]

# Dataset path (Make sure this path is correct)
data_dir = "dataset"
img_size = 128  
batch_size = 32  

# Data Augmentation to prevent overfitting
datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, rotation_range=30, zoom_range=0.2, horizontal_flip=True,
    validation_split=0.2  # Splitting dataset
)

# Training & Validation Data
train_data = datagen.flow_from_directory(
    data_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='sparse', subset='training')

val_data = datagen.flow_from_directory(
    data_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='sparse', subset='validation')

# Build CNN Model with Dropout
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),  # Dropout to reduce overfitting
    layers.Dense(len(categories), activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early Stopping Callback
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train Model
epochs = 15
history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[early_stopping])

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Training Performance')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Real-time Classification Function
def classify_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    predicted_label = categories[np.argmax(prediction)]
    
    return predicted_label

# Test Prediction
img_path = "test_sample2.jpg"  # Change this to a valid test image
print("Predicted Waste Category:", classify_image(img_path))

# Save Model
model.save("waste_classifier.h5")
