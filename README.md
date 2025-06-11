
# 🧠 Small Image Classification Using Convolutional Neural Network (CNN)

This project demonstrates image classification using a Convolutional Neural Network (CNN) on the CIFAR-10 dataset.

## 📚 Dataset
- CIFAR-10 from `tensorflow.keras.datasets`
- Contains 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## 🔧 Libraries Used
```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
```

## 📥 Data Loading
```python
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
```

- `X_train.shape`: `(50000, 32, 32, 3)`
- `y_train.shape`: `(50000, 1)`

## 🔍 Labels (Sample)
```python
y_train[:5] → [[6], [9], [9], [4], [1]]
y_test[:5]  → [[3], [8], [8], [0], [6]]
```

## 🛠️ Preprocessing
- Normalize pixel values: `X_train, X_test = X_train/255.0, X_test/255.0`
- Flatten `y_test` to 1D: `y_test = y_test.reshape(-1,)`

## 🏗️ CNN Model Architecture
```python
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])
```

## ⚙️ Compilation & Training
```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))
```

## 📈 Visualization
Plotting training and validation accuracy/loss over epochs using `matplotlib`.

## ✅ Evaluation
Final model evaluation on test data.
