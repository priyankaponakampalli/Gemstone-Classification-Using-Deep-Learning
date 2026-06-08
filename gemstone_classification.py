from google.colab import files
uploaded = files.upload()  # select archive.zip
import zipfile
with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/gemstones')

import os
print("Classes:", len(os.listdir('/content/gemstones/train')))
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception, ResNet50V2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# STEP 1: DATA PREPARATION (Better than paper)
# ============================================================
train_dir = '/content/gemstones/train'
test_dir = '/content/gemstones/test'

# Strong augmentation - better than paper
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

# 256x256 like paper
train_data = train_datagen.flow_from_directory(
    train_dir, target_size=(256,256),
    batch_size=32, class_mode='categorical',
    shuffle=True
)
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=(256,256),
    batch_size=32, class_mode='categorical',
    shuffle=False
)

print(f"Training images: {train_data.samples}")
print(f"Testing images: {test_data.samples}")
print(f"Number of classes: {train_data.num_classes}")

# ============================================================
# STEP 2: CALLBACKS (Extra improvement)
# ============================================================
def get_callbacks(model_name):
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f'{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

# ============================================================
# STEP 3: CNN MODEL (Improved from paper)
# ============================================================
def build_cnn():
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32,(3,3), activation='relu',
            padding='same', input_shape=(256,256,3)),
        layers.Conv2D(32,(3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64,(3,3), activation='relu', padding='same'),
        layers.Conv2D(64,(3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128,(3,3), activation='relu', padding='same'),
        layers.Conv2D(128,(3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Block 4 (extra - not in paper)
        layers.Conv2D(256,(3,3), activation='relu', padding='same'),
        layers.Conv2D(256,(3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(87, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ============================================================
# STEP 4: XCEPTION MODEL (Improved from paper)
# ============================================================
def build_xception():
    base = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(256,256,3)
    )
    # Freeze all first
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(87, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ============================================================
# STEP 5: RESNET50V2 MODEL (New contribution)
# ============================================================
def build_resnet():
    base = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(256,256,3)
    )
    # Freeze all first
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(87, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ============================================================
# STEP 6: TRAIN ALL 3 MODELS
# ============================================================

# Train CNN
print("\n" + "="*60)
print("           TRAINING CNN")
print("="*60)
cnn_model = build_cnn()
cnn_history = cnn_model.fit(
    train_data,
    epochs=25,
    validation_data=test_data,
    callbacks=get_callbacks('cnn')
)

# Train Xception
print("\n" + "="*60)
print("           TRAINING XCEPTION")
print("="*60)
xception_model = build_xception()
xception_history = xception_model.fit(
    train_data,
    epochs=25,
    validation_data=test_data,
    callbacks=get_callbacks('xception')
)

# Fine tune Xception (extra improvement)
print("\nFine tuning Xception...")
base = xception_model.layers[0]

for layer in base.layers[:-30]:
    layer.trainable = False

for layer in base.layers[-30:]:
    layer.trainable = True
xception_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
xception_ft_history = xception_model.fit(
    train_data,
    epochs=10,
    validation_data=test_data,
    callbacks=get_callbacks('xception_ft')
)

# Train ResNet50V2
print("\n" + "="*60)
print("           TRAINING RESNET50V2")
print("="*60)
resnet_model = build_resnet()
resnet_history = resnet_model.fit(
    train_data,
    epochs=25,
    validation_data=test_data,
    callbacks=get_callbacks('resnet')
)

# Fine tune ResNet (extra improvement)
print("\nFine tuning ResNet50V2...")
base = resnet_model.layers[0]

for layer in base.layers[:-30]:
    layer.trainable = False

for layer in base.layers[-30:]:
    layer.trainable = True
resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
resnet_ft_history = resnet_model.fit(
    train_data,
    epochs=10,
    validation_data=test_data,
    callbacks=get_callbacks('resnet_ft')
)

# ============================================================
# STEP 7: RESULTS
# ============================================================
cnn_acc = cnn_model.evaluate(test_data)[1] * 100
xception_acc = xception_model.evaluate(test_data)[1] * 100
resnet_acc = resnet_model.evaluate(test_data)[1] * 100

print("\n" + "="*50)
print("         FINAL RESULTS SUMMARY")
print("="*50)
print(f"CNN Accuracy:        {cnn_acc:.2f}%")
print(f"Xception Accuracy:   {xception_acc:.2f}%")
print(f"ResNet50V2 Accuracy: {resnet_acc:.2f}%")
print("="*50)

# ============================================================
# STEP 8: PLOTS FOR YOUR PAPER
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16,12))

# Accuracy bar chart
models_names = ['CNN', 'Xception', 'ResNet50V2']
accuracies = [cnn_acc, xception_acc, resnet_acc]
colors = ['#3498db', '#e74c3c', '#2ecc71']

axes[0,0].bar(models_names, accuracies, color=colors)
axes[0,0].set_title('Model Accuracy Comparison', fontsize=14)
axes[0,0].set_ylabel('Accuracy (%)')
axes[0,0].set_ylim([0, 100])
for i, v in enumerate(accuracies):
    axes[0,0].text(i, v+1, f'{v:.2f}%',
        ha='center', fontweight='bold')

# Training accuracy
axes[0,1].plot(cnn_history.history['val_accuracy'],
    label='CNN', color='#3498db')
axes[0,1].plot(xception_history.history['val_accuracy'],
    label='Xception', color='#e74c3c')
axes[0,1].plot(resnet_history.history['val_accuracy'],
    label='ResNet50V2', color='#2ecc71')
axes[0,1].set_title('Validation Accuracy over Epochs', fontsize=14)
axes[0,1].set_xlabel('Epochs')
axes[0,1].set_ylabel('Accuracy')
axes[0,1].legend()

# Training loss
axes[1,0].plot(cnn_history.history['val_loss'],
    label='CNN', color='#3498db')
axes[1,0].plot(xception_history.history['val_loss'],
    label='Xception', color='#e74c3c')
axes[1,0].plot(resnet_history.history['val_loss'],
    label='ResNet50V2', color='#2ecc71')
axes[1,0].set_title('Validation Loss over Epochs', fontsize=14)
axes[1,0].set_xlabel('Epochs')
axes[1,0].set_ylabel('Loss')
axes[1,0].legend()

# Training vs validation accuracy for ResNet
axes[1,1].plot(resnet_history.history['accuracy'],
    label='Train Accuracy', color='#2ecc71')
axes[1,1].plot(resnet_history.history['val_accuracy'],
    label='Val Accuracy', color='#e74c3c')
axes[1,1].set_title('ResNet50V2 Train vs Val Accuracy', fontsize=14)
axes[1,1].set_xlabel('Epochs')
axes[1,1].set_ylabel('Accuracy')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()
print("\nCharts saved for your paper!")
