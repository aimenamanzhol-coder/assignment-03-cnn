# Assignment 03: Convolutional Neural Networks

**Related to**: Week 5 - CNNs

## Part 1: CNN Forward Pass on Paper

You are given a grayscale image of size 6x6 and a simple CNN architecture. Your goal is to perform the forward pass on paper and calculate the output at each step of the network.

### Given CNN Architecture

1. Input: 6x6 grayscale image
2. Convolution Layer 1: 3x3 filter, stride 1, no padding, 2 filters (output: 4x4)
3. ReLU Activation
4. Max Pooling Layer: 2x2 filter, stride 2 (output: 2x2)
5. Flattening step
6. Linear layer into one output

### Steps to Complete

#### 1. Input Image
Start with a 6x6 matrix representing the grayscale image. Use the following matrix (pixel intensities):

```
[[1, 2, 0, 3, 1, 2],
 [2, 1, 1, 0, 2, 3],
 [0, 1, 2, 3, 1, 0],
 [3, 0, 1, 2, 2, 1],
 [2, 1, 0, 1, 3, 0],
 [1, 0, 2, 1, 0, 2]]
```

#### 2. Spatial Dimension Calculation
Calculate the output size for each layer using the formula: $(N - F + 2P) / S + 1$. Show your work for:
- Conv Layer 1
- Max Pooling Layer

#### 3. Convolution Layer
Apply 2 3x3 convolution filters (kernels) with the following weights and a bias of 1 for each filter:

**Filter 1:**
```
[[1, 0, -1],
 [1, 0, -1],
 [1, 0, -1]]
```

**Filter 2:**
```
[[-1, 2, -1],
 [0, 0, 0],
 [-1, 2, -1]]
```

- Use a stride of 1 and no padding
- Perform the convolution operation (Sum of element-wise multiplication + bias)
- Write down the resulting 4x4 feature maps

#### 4. ReLU Activation
Apply ReLU activation to the feature maps (replace negative values with 0).

#### 5. Max Pooling Layer
Perform max pooling with a 2x2 filter and a stride of 2.

#### 6. Parameter Calculation
Calculate the total number of trainable parameters in this network (include weights and biases for all layers).

## Part 2: PyTorch Implementation and Visualization of a CNN

**Objective**: Implement a custom CNN in PyTorch and visualize the feature maps after each layer to understand how the input image is transformed at every stage.

### Tasks

#### 1. Dataset Preparation

- Load the MNIST or CIFAR-10 dataset using PyTorch's `torchvision.datasets`
- Implement a preprocessing pipeline with `torchvision.transforms` including normalization and basic data augmentation (e.g., RandomHorizontalFlip)
- Split into training and testing sets

#### 2. Building the CNN

Define a simple CNN with the following layers:
- Input: Grayscale or RGB image
- Conv Layer 1: 3x3 filter, stride 1, padding 1, 16 filters
- ReLU Activation
- Max Pooling Layer: 2x2, stride 2
- Conv Layer 2: 3x3 filter, stride 1, padding 1, 32 filters
- ReLU Activation
- Max Pooling Layer: 2x2, stride 2
- Fully connected (FC) layer for classification

Print the model summary using `torchsummary`.

#### 3. Training the CNN

- Train the model on the training set using cross-entropy loss and Adam optimizer
- Track the training accuracy and loss over epochs

#### 4. Visualizing Feature Maps

Implement a function to visualize the feature maps at each stage of the CNN. After training, select a random test image and visualize:
- Output of Conv Layer 1
- Output after ReLU (Layer 1)
- Output after Max Pooling (Layer 1)
- Output of Conv Layer 2
- Output after ReLU (Layer 2)
- Output after Max Pooling (Layer 2)

**Note**: Use `matplotlib` or PIL to display the feature maps. Additionally, visualize the weights (kernels) of the first convolutional layer and discuss what patterns they appear to be detecting. Ensure dimensions are consistent and clearly label each visualization.

#### 5. Model Evaluation

- Test the trained model on the test set and report accuracy
- Visualize a few misclassified images and discuss why the model may have made mistakes

## Part 3: Questions for Understanding

Answer the following questions based on your work and CNN theory:

1. **Architecture & Design**:
   - Why do we typically use small kernels (like 3x3) instead of large ones (like 7x7 or 11x11) in modern architectures?
   - What is the difference between "Valid" and "Same" padding?

2. **Activation Functions**:
   - Why is ReLU preferred over Sigmoid in the hidden layers of a deep CNN?
   - What happens to the gradients when a ReLU unit outputs 0?

3. **Pooling**:
   - What is the primary purpose of Max Pooling? Does it have any trainable parameters?
   - How does pooling contribute to "translational invariance"?

4. **Training & Regularization**:
   - If your model has high training accuracy but low test accuracy, what is happening, and how would you fix it?
   - How does Dropout help prevent overfitting in a CNN?

5. **Theoretical Intuition**:
   - What features do the early layers of a CNN usually learn compared to the deeper layers?
   - Describe the concept of "Receptive Field" and how it changes as we go deeper into the network.
