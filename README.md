# Neural Cellular Automata

This project focuses on implementing Neural Cellular Automata (NCA) to learn unique 0-player game rulesets. The NCA model leverages convolutional neural networks (CNNs) to simulate the behavior of cellular automata, specifically the Game of Life.

## Introduction

Cellular Automata (CA) are mathematical models composed of a grid of cells that evolve over discrete time steps based on predefined rules. Each cell's state is determined by its own state and the states of its neighboring cells. The Game of Life, a well-known CA, demonstrates emergent behavior from simple rules.

In this project, we use Neural Cellular Automata, which combine CA with neural networks. Instead of explicitly defining the rules, the NCA model learns the rules through training. The goal is to train an NCA model that can accurately predict the state of a cell in the next time step based on its current state and its neighbors.

## Model Architecture

The NCA model architecture consists of the following components:

1. Input Layer: The input layer takes the initial state of the cells as input, represented as a grid of binary values (0 or 1).

2. Convolutional Layers: The convolutional layers apply filters to capture local patterns and features from the input. These layers help the model learn the spatial dependencies between cells.

3. Dense Layers: The dense layers perform computations on the extracted features and capture global patterns in the input.

4. Output Layer: The output layer produces the predicted state of each cell in the next time step. It uses the softmax activation function to convert the logits into probabilities.

## Training and Evaluation

The model is trained using a dataset of input-output pairs, where the input is the initial state of the cells, and the output is the next state of the cells. The loss function used for training is categorical cross-entropy, and the Adam optimizer is used to optimize the model parameters.

After training, the model is evaluated on a separate test dataset. The accuracy metric is used to measure the model's performance in predicting the next state of the cells.

## Usage

To use this project, follow these steps:

1. Install the required dependencies, such as TensorFlow, NumPy, and Matplotlib.

2. Create the training data by generating initial states for the cells and their corresponding next states using the `make_glider` function or other methods.

3. Build the NCA model by defining the model architecture using TensorFlow's Keras API.

4. Compile the model by specifying the optimizer, loss function, and metrics to be used during training.

5. Train the model using the training data and specified hyperparameters.

6. Evaluate the model's performance on a test dataset to assess its accuracy.

7. Save the trained model for future use.

8. Use the saved model to make predictions on new input data.

## Example

Here is an example code snippet that demonstrates the usage of this project:

```python
# Import necessary libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Create training data
train_size = 100
width = 10 
height = 10

# Generate initial states and corresponding next states
X_train = ...

# Define and build the NCA model
model = tf.keras.models.Sequential(...)
model.compile(...)

# Train the model
model.fit(...)

# Evaluate the model on a test dataset
test_size = 100
X_test = ...
Y_test = ...

eval = model.evaluate(X_test, Y_test)
print(eval)

# Make predictions with the trained model
X_new = ...
predictions = model.predict(X_new)

# Save the trained model
model.save('saved_model/my_model')
```

## Conclusion

Neural Cellular Automata provide a powerful framework for learning rulesets of 0-player games. By combining the dynamics of cellular automata with the capabilities of neural networks, we can train models to capture complex patterns and generate accurate predictions about the next state of the cells. This project serves as a starting point for exploring various game rulesets and training NCAs to learn and simulate their behavior.
