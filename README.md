<div align="center">

# 🧠 Deep Learning — From Perceptrons to RNNs

### Following [CampusX: 100 Days of Deep Learning](https://www.youtube.com/playlist?list=PLKnIA16_Rmvb-Ub83D8K-xsM2JF7A4A-U)

![Progress](https://img.shields.io/badge/Progress-Active-brightgreen?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20PyTorch-orange?style=flat-square)
![Language](https://img.shields.io/badge/Language-Python-blue?style=flat-square)
![Course](https://img.shields.io/badge/Course-CampusX%20100%20Days%20of%20DL-purple?style=flat-square)

> *Building deep learning from first principles — one neuron at a time.*

</div>

---

## What This Is

A hands-on deep learning journey — every notebook here was run personally. The focus is on understanding what happens *underneath* the frameworks: backpropagation derived by hand, weight initialization, RNNs built from scratch, and PyTorch pipelines.

The repository is organized **by domain**, so each folder groups related experiments, its datasets, and its visual assets together.

---

## Repository Layout

```
DL/
├── perceptron/               # Single neuron: perceptron, hinge loss
├── ann-fundamentals/         # Backprop, optimizers, weight init, regularization, dropout, batch-norm, early stopping
├── pytorch-pipelines/        # PyTorch from tensors → autograd → complete training loops
├── ann-applications/         # Real datasets: GRE, insurance, churn, breast cancer, MNIST
├── cnn/                      # Convolutions, data augmentation, transfer learning
├── rnn/                      # RNNs from scratch, Elman/Jordan/Hopfield experiments
├── lstm-gru/                 # LSTM, deep LSTM, GRU, text generation
├── nlp/                      # Tokenization, filtering, stemming & lemmatization
├── hyperparameter-tuning/    # Keras Tuner notebooks
├── references/               # Colab-era course notebooks (reference only)
├── notes/                    # Learning notes & discoveries
└── README.md
```

---

## 📂 Perceptron

The beginning — a single neuron.

| File | Topic |
|---|---|
| `PerceptronFromScratch.ipynb` | Perceptron implemented from scratch |
| `perceptron-using-hinge-loss.ipynb` | Hinge loss and the perceptron learning rule |
| `perceptron-inution.ipynb` | Intuition behind the perceptron |

---

## 📂 Ann Fundamentals

The math and training dynamics that make ANNs actually work.

| File | Topic |
|---|---|
| `backpropagation_from_scratch_for_classification.ipynb` | Backprop for classification (cross-entropy, sigmoid, softmax) |
| `backpropogation_from_scratch_regression.ipynb` | Backprop for regression (MSE) |
| `gradient_descent_in_nn.ipynb` | Batch / Stochastic / Mini-batch gradient descent |
| `vanishing-gradient-problem-and-solutions.ipynb` | Why sigmoid/tanh vanish, ReLU as a fix |
| `strategies-for-weight-initialization.ipynb` | Xavier, He initialization |
| `wrong-strategies-of-weight-initialization.ipynb` | What goes wrong with bad init |
| `feature-scalling-and-its-affect-in-ann.ipynb` | Effect of unscaled features on training |
| `regularization-in-ann.ipynb` | L1 / L2 regularization |
| `dropout-on-regression.ipynb` · `dropout-on-classification.ipynb` | Dropout on both task types |
| `batch-normalization-on-ann.ipynb` | Batch normalization internals |
| `early-stoping-in-nn.ipynb` | Early stopping to prevent overfitting |

**Assets:** `ann-visualizations/` (architecture diagrams, matrix math, optimizer trajectories) · `optimizers-visualization/` (animated SGD, Momentum, RMSProp, Adam) · datasets `Social_Network_Ads.csv`, `artificial-dataset-classification.csv`

---

## 📂 PyTorch Pipelines

From raw tensors to a production-style training loop — all in PyTorch.

| File | Topic |
|---|---|
| `intro-to-torch.ipynb` | PyTorch introduction |
| `tensors_in_pytorch.ipynb` | Tensor creation, properties, indexing |
| `autograd_in_torch.ipynb` | Autograd vs. manual gradient computation |
| `pipeline_using_nn_and_optim_module_using_torch.ipynb` | NN pipeline using `nn` + `optim` |
| `manual_practical_pipeline_using_torch.ipynb` | Manual network without `nn`/`optim` |
| `Dataset_and_DataLoader_class_in_pytorch.ipynb` | Dataset & DataLoader abstractions |
| `complete_training_pipeline_on_breast_cancer_dataset_in_torch.ipynb` | End-to-end training loop |

---

## 📂 Ann Applications

Applying ANNs to real tabular datasets. Datasets are bundled with the notebooks.

| File | Dataset | Task |
|---|---|---|
| `gre-admission-using-ann.ipynb` | `Admission_Predict_Ver1.1.csv` | Regression |
| `neural-network-on-insurance-dataset.ipynb` | `insurance.csv` | Regression |
| `neural-network-on-insurance-dataset-copy.ipynb` | `insurance.csv` | Regression |
| `customer-churn-prediction-using-ANN.ipynb` | `Churn_Modelling.csv` | Classification |
| `BreastCancerClassifierUsingANN.ipynb` | `load_breast_cancer` (sklearn) | Classification |
| `mnist_classification_using_ann.ipynb` | MNIST | Multi-class Classification |

*(`bank-additional-full.csv` is also included here as a spare dataset.)*

---

## 📂 CNNs

Convolutions, augmentation, and transfer learning.

| File | Topic |
|---|---|
| `intro-to-cnns.md` | How CNNs work — notes |
| `convoltional-neural-network-basics.ipynb` | Filters, feature maps, pooling from scratch |
| `cnn_using_pytorch.ipynb` | Convolution/ReLU/pooling in PyTorch |
| `cnn-data-augementation-on-images.ipynb` | Augmentation pipeline on cats vs dogs |
| `catsvsdogs-cnn-overfits.ipynb` | Overfitting on small image datasets |
| `catsvsdogs-resnet50-vs-vgg16.ipynb` | Transfer learning: ResNet50 vs VGG16 |

**Assets:** `cnn-architecture/` (LeNet, VGG, ResNet diagrams) · `cnn-visualizations/` · `augmented_output/` · `train/` (source image)

---

## 📂 RNNs

The birth of sequence modeling — from scratch.

| File | Topic |
|---|---|
| `birth-of-rnn.md` | Why ANNs failed and RNNs were born — notes |
| `rnn_from_scratch.ipynb` | RNN forward/backprop built from scratch |
| `simple_rnn_using_pytorch.ipynb` | Simple RNN in PyTorch |
| `elman_experiment_on_rnn_using_pytorch.ipynb` | Elman's clustering experiment |
| `jordan_experiment_rnn.ipynb` | Jordan's contextual layer experiment |
| `hopfield-experiment-for-rnn.ipynb` | Hopfield memory network |
| `rnn_on_nlp_using_keras.ipynb` | RNN on NLP with Keras |
| `SentimentAnalysisForRNNusingkeras.ipynb` | Sentiment analysis with Keras RNN |

**Assets:** `rnn-architecture/` (diagrams)

---

## 📂 LSTM / GRU

Gated architectures and text generation.

| File | Topic |
|---|---|
| `lstm-using-pytorch.ipynb` | LSTM for sentiment classification |
| `deeplstm-using-pytorch.ipynb` | 3-layer deep LSTM |
| `gru_using_pytorch.ipynb` | GRU comparison + LR scheduling |
| `next-word-predictor-using-lstm.ipynb` | Next-word prediction (Wikitext) |
| `char_generation_simple.ipynb` · `char_generation_rnn.py` | Character-level text generation (Shakespeare) |

**Data:** `imdb/` (Train/Valid/Test), `wikitext_train.txt`, `shakespeare.txt`

---

## 📂 NLP

Text preprocessing scripts.

| File | Topic |
|---|---|
| `tokenization-filtering-and-script-detection.py` | Tokenization, stopword filtering, script detection |
| `stemming_and_lemmatization.py` | Stemming vs. lemmatization |

**Assets:** `NLP_experiments/` (experiment write-ups & results)

---

## 📂 Hyperparameter Tuning

Systematic search with Keras Tuner — neurons, layers, activation functions, learning rate.

```
hyperparameter-tuning-using-keras-tuner-on-hyperparameters/
├── tuning-no-of-neurons.ipynb
├── tuning-no-of-neurons-and-no-of-layers.ipynb
├── tuning-...-and-activation-function.ipynb
└── complete-hyperparameter-tuning.ipynb
```

---

## 📂 Notes

Learning notes and discoveries:

- `discoveries/` — why DL, CPU → GPU → TPU evolution, NLP notes
- `AST_Document.svg`, `cmaps.txt`, `image.png`

---

## 📂 References

Colab-era course notebooks from CampusX, kept for reference only. These still use `/content/` paths and are not maintained.

---

## Stack

```python
Python 3.x
PyTorch · TensorFlow / Keras
NumPy · Pandas · Matplotlib · Scikit-learn
NLTK · Keras Tuner
```

---

<div align="center">

*Part of the [CampusX 100 Days of Deep Learning](https://www.youtube.com/playlist?list=PLKnIA16_Rmvb-Ub83D8K-xsM2JF7A4A-U) series*

</div>
