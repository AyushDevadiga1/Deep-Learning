A **Convolutional Neural Network (CNN)** is a deep learning architecture designed primarily to process spatial data, such as images. Unlike traditional neural networks that treat images as flat vectors of numbers, CNNs preserve the spatial structure (height, width, channels) to detect patterns like edges, textures, shapes, and complex objects.

Here is the complete step-by-step workflow of a CNN model architecture, followed by a concrete numerical example.

---

![alt text](image.png)

---

## 1. Core Workflow & Architectural Layers

A typical CNN processes images through a pipeline of feature extraction layers followed by classification layers.

### Step 1: Input Layer

The network receives raw pixel values as a multi-dimensional array (tensor).

* **Grayscale Image:** $H \times W \times 1$ (Height, Width, 1 Color Channel)
* **Color Image (RGB):** $H \times W \times 3$ (Height, Width, 3 Color Channels)

---

### Step 2: Convolutional Layer (Feature Extraction)

The core building block of a CNN. A small matrix of learnable weights called a **filter** (or **kernel**) slides across the input image to calculate local dot products, producing a 2D **feature map**.

* **Kernel / Filter:** Smaller grid (e.g., $3 \times 3$ or $5 \times 5$) that detects specific features (edges, curves, gradients).
* **Stride ($S$):** The number of pixels the filter moves at each step.
* **Padding ($P$):** Zero-padding added to the borders of the image to preserve spatial dimensions.

> **Output Dimension Formula:**
> Given an input size $W$, kernel size $K$, padding $P$, and stride $S$, the output dimension $O$ for spatial size is:
> 
> $$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
> 
> 

---

### Step 3: Activation Function (Non-Linearity)

Because convolution is a linear operation, an activation function introduces non-linearity, allowing the model to learn complex patterns.

* The standard choice is **ReLU (Rectified Linear Unit)**:

$$f(x) = \max(0, x)$$


* It converts all negative pixel values in the feature map to zero while keeping positive values unchanged.

---

### Step 4: Pooling Layer (Downsampling)

Pooling reduces the spatial size (width and height) of feature maps while preserving critical features. This reduces computation and helps make feature detection invariant to small translations and distortions.

* **Max Pooling:** Selects the maximum value from each patch (most common).
* **Average Pooling:** Takes the average value of each patch.

---

### Step 5: Flattening Layer

Once feature extraction is complete, the 3D tensor output from the final pooling layer is "flattened" into a single 1D vector. This vector serves as the bridge between spatial feature extraction and final classification.

---

### Step 6: Fully Connected (Dense) & Output Layers

* **Fully Connected (FC) Layer:** Connects every neuron in the vector to every neuron in the next layer, learning high-level combinations of features.
* **Output Layer:** Uses an activation function to generate final predictions:
* **Softmax:** Used for multi-class classification (outputs probability distribution summing to 1).
* **Sigmoid:** Used for binary classification (outputs a probability between 0 and 1).



---

## 2. Detailed Step-by-Step Example

Let's trace a sample RGB image passing through a simple classification network.

### Model Goal

Classify a $32 \times 32 \times 3$ color image into one of **10 object classes** (e.g., CIFAR-10 classification).

---

### Step-by-Step Dimensional Tracking

| Layer Stage | Layer Type | Specifications / Parameters | Math Calculation | Output Shape |
| --- | --- | --- | --- | --- |
| **0. Input** | Input Image | $32 \times 32 \times 3$ RGB Image | N/A | **$32 \times 32 \times 3$** |
| **1. Conv 1** | Convolution | 32 filters, $3 \times 3$ kernel, $P=1, S=1$ | $O = \frac{32 - 3 + 2(1)}{1} + 1 = 32$ | **$32 \times 32 \times 32$** |
| **2. ReLU 1** | Activation | Element-wise $\max(0, x)$ | Same shape | **$32 \times 32 \times 32$** |
| **3. Pool 1** | Max Pooling | $2 \times 2$ pool size, $S=2$ | $O = \frac{32}{2} = 16$ | **$16 \times 16 \times 32$** |
| **4. Conv 2** | Convolution | 64 filters, $3 \times 3$ kernel, $P=1, S=1$ | $O = \frac{16 - 3 + 2(1)}{1} + 1 = 16$ | **$16 \times 16 \times 64$** |
| **5. Pool 2** | Max Pooling | $2 \times 2$ pool size, $S=2$ | $O = \frac{16}{2} = 8$ | **$8 \times 8 \times 64$** |
| **6. Flatten** | Reshape | Combine spatial dimensions into 1D | $8 \times 8 \times 64 = 4096$ | **$4096$** |
| **7. Dense 1** | Fully Connected | 128 neurons + ReLU activation | $4096 \rightarrow 128$ | **$128$** |
| **8. Output** | Dense + Softmax | 10 output classes | $128 \rightarrow 10$ | **$10$** |

---

## Summary of Data Flow

$$\text{Input } (32 \times 32 \times 3) \xrightarrow{\text{Conv1 + ReLU}} (32 \times 32 \times 32) \xrightarrow{\text{MaxPool1}} (16 \times 16 \times 32)$$

$$\xrightarrow{\text{Conv2 + ReLU}} (16 \times 16 \times 64) \xrightarrow{\text{MaxPool2}} (8 \times 8 \times 64) \xrightarrow{\text{Flatten}} (4096)$$

$$\xrightarrow{\text{Dense}} (128) \xrightarrow{\text{Softmax Output}} (10 \text{ Probabilities})$$