<div align="center">

# 🧠 Deep Learning — From Scratch to CNNs

### Following [CampusX: 100 Days of Deep Learning](https://www.youtube.com/playlist?list=PLKnIA16_Rmvb-Ub83D8K-xsM2JF7A4A-U)

![Progress](https://img.shields.io/badge/Progress-Active-brightgreen?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange?style=flat-square)
![Language](https://img.shields.io/badge/Language-Python-blue?style=flat-square)
![Course](https://img.shields.io/badge/Course-CampusX%20100%20Days%20of%20DL-purple?style=flat-square)

> *Building deep learning from first principles — one neuron at a time.*

</div>

---

## What This Is

This repository tracks my hands-on journey through deep learning, following the **CampusX 100 Days of Deep Learning** series. Every notebook here is something I ran myself — the goal is not just to use tools, but to understand what's happening underneath them.

Topics range from the mathematics of a single perceptron all the way up to convolutional neural networks, transfer learning, and hyperparameter tuning.

---

## Topics Covered

### 1. Foundations of Deep Learning

**Why Deep Learning?**
- The evolution from CPUs → GPUs → TPUs and why compute changed everything
- What makes DL different from classical ML — representation learning vs. feature engineering

**The Perceptron**
- Building a perceptron from scratch in Python
- Perceptron on linearly separable vs. non-linear data
- Hinge loss and the perceptron learning rule

📓 `perceptron.ipynb` · `PerceptronFromScratch.ipynb` · `perceptron-using-hinge-loss.ipynb`

---

### 2. Neural Network Architecture & Math

**Structure and Notation**
- Layers, neurons, weights, biases — the full notation system
- Matrix math behind a forward pass
- How the architecture shape determines capacity

**Forward Propagation**
- Step-by-step computation through a multi-layer network
- Vectorized implementation using NumPy

📓 `intro_to_DL.txt` · `neural_network_architecture_shape.svg` · `nn_matrix_math_clean.svg` · `nn_full_summary_black.svg`

---

### 3. Backpropagation (From Scratch)

One of the most important sections — implementing backprop without any framework, just calculus and NumPy.

- Deriving gradients through the chain rule
- Backprop for **regression** — MSE loss, weight updates
- Backprop for **classification** — cross-entropy, sigmoid, softmax
- Side-by-side comparison of regression vs. classification backprop

📓 `backpropagation_from_scratch_for_classification.ipynb`
📓 `backpropogation_from_scratch_regression.ipynb`
📝 `backprop_regression_vs_classification.svg`

---

### 4. Training Dynamics

**Gradient Descent Variants**
- Batch GD, Stochastic GD, Mini-batch GD — when to use which
- Visualized optimizer paths: SGD · Momentum · RMSProp · Adam (stable & unstable)

**Weight Initialization**
- Why random initialization matters
- Wrong strategies and the problems they cause (vanishing/exploding activations)
- Correct strategies: Xavier, He initialization

📓 `gradient_descent_in_nn.ipynb`
📓 `strategies-for-weight-initialization.ipynb`
📓 `wrong-strategies-of-weight-initialization.ipynb`
🎞️ `optimizers-visualization/` — animated GIFs for each optimizer

---

### 5. Making ANNs Actually Work

**Feature Scaling**
- How unscaled features wreck gradient descent in ANNs
- Min-max vs. standardization — effects on training curves

**Activation Functions & Vanishing Gradients**
- The vanishing gradient problem with sigmoid/tanh
- ReLU and its variants as solutions

**Regularization**
- L1/L2 regularization in ANNs
- Dropout on regression and classification tasks

**Batch Normalization**
- Internal covariate shift
- BN layer placement and effect on training speed

**Early Stopping**
- Monitoring val loss to halt overfitting automatically

📓 `feature-scalling-and-its-affect-in-ann.ipynb`
📓 `vanishing-gradient-problem-and-solutions.ipynb`
📓 `regularization-in-ann.ipynb`
📓 `dropout-on-regression.ipynb` · `dropout-on-classification.ipynb`
📓 `batch-normalization-on-ann.ipynb`
📓 `early-stoping-in-nn.ipynb`

---

### 6. ANN on Real Datasets

Applying everything above to real-world problems:

| Notebook | Dataset | Task |
|---|---|---|
| `gre-admission-using-ann.ipynb` | `Admission_Predict_Ver1.1.csv` | Regression |
| `neural-network-on-insurance-dataset.ipynb` | `insurance.csv` | Regression |
| `customer-churn-prediction-using-ANN.ipynb` | `Churn_Modelling.csv` | Classification |
| `mnist_classification_using_ann.ipynb` | MNIST (28×28 images) | Multi-class Classification |
| `perceptro-on-non-linear-data.png` | `artificial-dataset-classification.csv` | Classification |

---

### 7. Convolutional Neural Networks (CNNs)

**How Convolution Works**
- Filters, feature maps, receptive fields
- Edge detection with manual convolution kernels
- Max pooling and spatial downsampling

**CNN Architecture**
- Full conv → pool → flatten → dense pipeline
- Dimension flow through each layer visualized

**Applying CNNs**
- MNIST with a CNN (vs. plain ANN — big accuracy jump)
- Cats vs. Dogs — where CNNs start to shine
- Overfitting on small image datasets and why it happens

📓 `convoltional-neural-network-basics.ipynb`
📓 `catsvsdogs-cnn-overfits.ipynb`
📝 `cnn_dimension_flow.png` · `convoltional-neural-network-architecture.jpg`

---

### 8. Data Augmentation

- Flips, rotations, zoom, shear — expanding a small dataset artificially
- Augmentation pipeline built with Keras `ImageDataGenerator`
- Visual output of augmented samples saved in `augmented_output/`

📓 `cnn-data-augementation-on-images.ipynb`

---

### 9. Transfer Learning

Using pre-trained ImageNet weights instead of training from scratch:

| Model | Architecture | Use |
|---|---|---|
| VGG16 | Deep stack of 3×3 convs | Feature extraction baseline |
| ResNet50 | Skip connections (residual blocks) | Better gradient flow, deeper |

- Head-to-head comparison: VGG16 vs. ResNet50 on Cats vs. Dogs
- Fine-tuning vs. feature extraction

📓 `catsvsdogs-resnet50-vs-vgg16.ipynb`
📝 `cnn-architecture/` — architecture diagrams for LeNet, VGG, ResNet

---

### 10. Hyperparameter Tuning with Keras Tuner

Systematic search instead of guessing:

| Notebook | What's Tuned |
|---|---|
| `tuning-no-of-neurons.ipynb` | Number of neurons per layer |
| `tuning-no-of-neurons-and-no-of-layers.ipynb` | Neurons + depth |
| `tuning-...-and-activation-function.ipynb` | Neurons + depth + activation |
| `complete-hyperparameter-tuning.ipynb` | Neurons + depth + activation + learning rate |

📓 All in `hyperparameter-tuning-using-keras-tuner-on-hyperparameters/`

---

## Folder Structure

```
DL/
├── optimizers-visualization/       # Animated optimizer comparisons (GIFs)
├── cnn-architecture/               # LeNet, VGG, ResNet diagrams
├── hyperparameter-tuning.../       # Keras Tuner notebooks (4 levels)
├── augmented_output/               # Sample augmented cat images
├── train/                          # Raw training images
│
├── perceptron.ipynb                # The beginning — single neuron
├── PerceptronFromScratch.ipynb
├── perceptron-using-hinge-loss.ipynb
│
├── backpropagation_from_scratch_for_classification.ipynb
├── backpropogation_from_scratch_regression.ipynb
├── gradient_descent_in_nn.ipynb
│
├── vanishing-gradient-problem-and-solutions.ipynb
├── strategies-for-weight-initialization.ipynb
├── wrong-strategies-of-weight-initialization.ipynb
├── feature-scalling-and-its-affect-in-ann.ipynb
│
├── regularization-in-ann.ipynb
├── dropout-on-regression.ipynb
├── dropout-on-classification.ipynb
├── batch-normalization-on-ann.ipynb
├── early-stoping-in-nn.ipynb
│
├── gre-admission-using-ann.ipynb
├── neural-network-on-insurance-dataset.ipynb
├── customer-churn-prediction-using-ANN.ipynb
├── mnist_classification_using_ann.ipynb
│
├── convoltional-neural-network-basics.ipynb
├── cnn-data-augementation-on-images.ipynb
├── catsvsdogs-cnn-overfits.ipynb
├── catsvsdogs-resnet50-vs-vgg16.ipynb
│
├── max-pooling-on-mnist.png
├── backprop_regression_vs_classification.svg
├── why_dl_breakthrough.svg
└── ... (datasets, diagrams, assets)
```

---

## Key Takeaways So Far

- Backpropagation is just the chain rule applied recursively — implementing it from scratch makes gradient flow intuitive, not magical.
- Vanishing gradients are a *geometry* problem: sigmoid squashes everything into [0,1], and chained multiplications go to zero. ReLU simply doesn't.
- CNNs work because natural images have local structure and translation invariance — convolution exploits both.
- Data augmentation is regularization in disguise — it's not adding new data, it's constraining the hypothesis space.
- Transfer learning is almost always the right first move on small image datasets.

---

## Stack

```python
Python 3.x
TensorFlow / Keras
NumPy · Pandas · Matplotlib · Scikit-learn
Keras Tuner
```

---

<div align="center">

*Part of the [CampusX 100 Days of Deep Learning](https://www.youtube.com/playlist?list=PLKnIA16_Rmvb-Ub83D8K-xsM2JF7A4A-U) series*

</div>