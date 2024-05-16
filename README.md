# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
In this experiment, we use an autoencoder to process handwritten digit images from the MNIST dataset. The autoencoder learns to encode and decode the images, reducing noise through layers like MaxPooling and convolutional. Then, we repurpose the encoded data to build a convolutional neural network for classifying digits into numerical values from 0 to 9. The goal is to create an accurate classifier for handwritten digits removing noise.
## Convolution Autoencoder Network Model

![image](https://github.com/Yuvan291205/convolutional-denoising-autoencoder/assets/138849170/3da2f41b-911b-4e02-bf82-ae3af0b8e3b9)


## DESIGN STEPS
### STEP 1:
Import the necessary libraries and load the mnist dataset without label column (y).

### STEP 2:
Scale the input (gray scale) images between 0 to 1.

### STEP 3:
Add noise to the input image and scale the noised image between 0 and 1.

### STEP 4:
Build the Neural Network model with

Encoder,
Convolutional layer,
Max Pooling (downsampling) layer,
Decoder,
Convolutional layer,
Upsampling layer.

### STEP 5:
Compile and fit the model.

### STEP 6:
Plot the predictions.

## PROGRAM
#### NAME: LAAKSHIT D
#### REG NO. 212222230071
```py
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
```
```py
(x_train, _), (x_test, _) = mnist.load_data()

x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
```
```py
n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```
#### Noicy image

![image](https://github.com/laakshit-D/convolutional-denoising-autoencoder/assets/119559976/eabdbb1b-5c21-4885-8a95-7903b663c7c9)

```py
input_img = keras.Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)

# Encoder output dimension is (7, 7, 32)

# Decoder
x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
```
```py
autoencoder.summary()
```

![image](https://github.com/laakshit-D/convolutional-denoising-autoencoder/assets/119559976/653c6841-70c3-464e-9afa-e2fb8a062348)

```py
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history = autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```
```
decoded_imgs = autoencoder.predict(x_test_noisy)
```
```py
n = 10

print("Developed by LAAKSHIT D (212222230071)")
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/laakshit-D/convolutional-denoising-autoencoder/assets/119559976/b5b3801a-7bf7-4a29-b66b-8a53a20a6ee8)

### Original vs Noisy Vs Reconstructed Image

![image](https://github.com/laakshit-D/convolutional-denoising-autoencoder/assets/119559976/d245fa13-600b-4790-8916-e61602ad0dbc)

## RESULT
Thus we have successfully developed a convolutional autoencoder for image denoising application.
