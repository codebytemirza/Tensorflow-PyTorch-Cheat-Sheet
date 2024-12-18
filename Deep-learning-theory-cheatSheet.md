# **Cheat Sheet of Deep Learning Theory**  
**By Muhammad Abdullah (Generated by AI)**

![Abdullah Mirza AI and Machine Learning Developer](https://media.licdn.com/dms/image/v2/D4D16AQEOrUjAALRlBw/profile-displaybackgroundimage-shrink_350_1400/profile-displaybackgroundimage-shrink_350_1400/0/1728131435697?e=1736985600&v=beta&t=qXMjqAkCNagH3IYRfixQ6znlZAZ4P-qXsV6Bf16Jr28)
---

## **Table of Contents**
1. [Introduction to Deep Learning](#1-introduction-to-deep-learning)
2. [Neural Networks: Perceptron and ANN](#3-neural-networks-perceptron-and-ann)
3. [Activation Functions](#4-activation-functions)
4. [Forward Propagation and Backpropagation](#5-forward-propagation-and-backpropagation)
5. [Optimization Algorithms](#6-optimization-algorithms)
6. [Neural Network Architectures: FNN, CNN, RNN](#7-neural-network-architectures-fnn-cnn-rnn)
7. [Code Examples: FNN, CNN, RNN](#8-code-examples-fnn-cnn-rnn)

---

## **1. Introduction to Deep Learning**

### What is Deep Learning?
Deep learning is a subset of machine learning where models learn to make decisions from large amounts of data using artificial neural networks (ANNs). These networks have many layers, making them "deep," and they are capable of learning complex patterns.

### Why Use Deep Learning?
- **Image Recognition**: Like identifying objects in pictures.
- **Speech Recognition**: Converting spoken language into text.
- **Natural Language Processing (NLP)**: Understanding and generating human language.
- **Time-Series Prediction**: Forecasting future values based on historical data.

### When to Use Deep Learning?
- When you have a **lot of data** and **computing power**.
- Tasks that involve complex data such as images, speech, and language.

---

## **2. Neural Networks: Perceptron and ANN**

### **Perceptron**:
A **Perceptron** is the simplest form of a neural network used for binary classification. It takes an input, applies a weight, adds a bias, and then outputs a result using an activation function.

**Perceptron Formula**:
\[
y = f(w_1x_1 + w_2x_2 + \dots + w_nx_n + b)
\]
Where:
- \(x_1, x_2, \dots\) are inputs.
- \(w_1, w_2, \dots\) are weights.
- \(b\) is the bias.
- \(f\) is the activation function.

### **Artificial Neural Networks (ANN)**:
ANNs consist of layers:
- **Input layer**: Takes in features.
- **Hidden layers**: Where the computation happens.
- **Output layer**: Produces the final result.

**How ANN Works**:
- Each neuron receives inputs, multiplies them by weights, adds a bias, and then applies an activation function.

## What are ANNs?

![alt text](https://cdn-images-1.medium.com/max/1200/1*-teDpAIho_nzNShRswkfrQ.gif)

An ANN is made of many interconnected "**neurons**".   

![alt text](https://www.softvision.com/wp-content/uploads/2019/01/Neural_Networks_in_your_browser_1.gif)

Each neuron takes in some floating point numbers (e.g. 1.0, 0.5, -1.0) and multiplies them by some other floating point numbers (e.g. 0.7, 0.6, 1.4) known as **weights** (1.0 * 0.7 = 0.7, 0.5 * 0.6 = 0.3, -1.0 * 1.4 = -1.4).  The weights act as a mechanism to focus on, or ignore, certain inputs.  The weighted inputs then get summed together (e.g. 0.7 + 0.3 + -1.4 = -0.4) along with a **bias** value (e.g. -0.4 + ** -0.1 ** = -0.5).  

The summed value (x) is now transformed into an output value (y) according to the neuron's **activation function**  (y = **f**(x)).  Some popular activation functions are shown below: 

![alt text](https://cdn-images-1.medium.com/max/1600/1*RD0lIYqB5L2LrI2VTIZqGw.png)

e.g. -0.5 --> -0.05 if we use the **Leaky Rectified Linear Unit (Leaky ReLU)** activation function: y = f(x) = f(-0.5) = max(0.1*-0.5, -0.5) = max(-0.05, -0.5) = -0.05

In larger ANNs with many layers, the neuron's output value (e.g. -0.05) would become the input for another neuron.

![alt text](https://www.neuraldesigner.com/images/deep_neural_network.png)

However, one of the first ANNs was known as the perceptron and it consisted of only a single neuron.  

![alt text](https://cdn-images-1.medium.com/max/1600/1*_Zy1C83cnmYUdETCeQrOgA.png)

The output of the perceptron's neuron acts as the final prediction. 

![alt text](https://cdn-images-1.medium.com/fit/t/1600/480/1*gpH4JC6Dqx_hIjrrcrq1Og.gif)

This means that each neuron is a linear binary classifier all on its own (e.g. an output value >= 0 would be the blue class, but an output value < 0 would be the red class)

---

## **3. Activation Functions**

Activation functions decide how much signal a neuron will pass on to the next layer.

### Types of Activation Functions:
1. **ReLU (Rectified Linear Unit)**:
   - **Formula**:  
     \[
     f(x) = \max(0, x)
     \]
   - **Purpose**: If input is positive, it stays the same. If negative, it becomes zero. Great for hidden layers because it avoids the vanishing gradient problem.

2. **Sigmoid**:
   - **Formula**:  
     \[
     f(x) = \frac{1}{1 + e^{-x}}
     \]
   - **Purpose**: Outputs values between 0 and 1. Used for binary classification.

3. **Softmax**:
   - **Formula**:  
     \[
     f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
     \]
   - **Purpose**: Turns raw outputs into probabilities for multi-class classification.

---

## **4. Forward Propagation and Backpropagation**

### **Forward Propagation**:
Forward propagation is the process of passing input through the network, layer by layer, to produce an output. Each layer performs computations using the inputs, weights, and biases.

### **Backpropagation**:
Backpropagation is used to train the network. It adjusts the weights to minimize the error by:
1. **Calculating the gradient of the error** with respect to each weight.
2. **Updating weights** using an optimization algorithm (like Gradient Descent).

Backpropagation helps in learning by correcting the model after every prediction.

---

## **5. Optimization Algorithms**

### What is Optimization?
Optimization refers to the process of minimizing (or maximizing) a function. In deep learning, we aim to **minimize the loss function** by adjusting the weights of the neural network.

### Common Optimization Algorithms:
1. **Gradient Descent**:
   - **Purpose**: Moves in the direction of the steepest decrease in the loss function.
   - **When to Use**: Basic and most common optimizer.

2. **Adam**:
   - **Purpose**: Combines the benefits of **Momentum** and **RMSProp**, adjusting learning rates for each parameter.
   - **When to Use**: Preferred in complex models due to efficiency.

3. **RMSProp**:
   - **Purpose**: Adapts the learning rate based on recent gradient updates, helping with noisy gradients.
   - **When to Use**: Works well with non-stationary data.

---

## **6. Neural Network Architectures: FNN, CNN, RNN**

### **Feedforward Neural Networks (FNN)**:
- **Description**: The simplest type of neural network where information flows from input to output in one direction.
- **Use Cases**: General-purpose tasks like classification and regression.

### **Convolutional Neural Networks (CNN)**:
- **Description**: Designed for processing images. Uses **convolution layers** to detect patterns like edges or textures in images.
- **Use Cases**: Image classification, object detection, and facial recognition.

### **Recurrent Neural Networks (RNN)**:
- **Description**: Designed to handle sequential data. RNNs have **memory** that allows them to use information from previous steps to influence the current step.
- **Use Cases**: Time series prediction, speech recognition, and language modeling.

---

## **7. Code Examples: FNN, CNN, RNN**

### **Feedforward Neural Network (FNN) Example**:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Simple Feedforward Neural Network
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_size,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # For binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)
```

### **Convolutional Neural Network (CNN) Example**:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Simple CNN for image classification
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes for multi-class classification
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)
```

### **Recurrent Neural Network (RNN) Example (LSTM)**:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# LSTM for time series forecasting
model = Sequential([
    LSTM(50, activation='relu', input_shape=(time_steps, features)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=10)
```
---

## **Refrence Link**
[What is ANN](https://colab.research.google.com/github/mohammedterry/ANNs/blob/master/ML_ANN.ipynb)
