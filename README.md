# Music Genre Classifier — AP Calculus BC Final Project

A feedforward neural network built from scratch in Python that listens to a song and predicts its genre. Trained on 1,000 audio clips across 10 genres, it reaches **75% test accuracy** — and every piece of math inside it connects directly to AP Calculus BC.

> *"A 75.5% accurate model that can be explained mathematically is more valuable than a more accurate one that cannot be explained at all."*
> — from the written report

***

## The Idea

For the AP Calculus BC "Just Because" final project, the goal was to learn something genuinely useful before studying Data Science in college — and to find the calculus hidden inside it. Neural networks turned out to be the perfect choice: under the hood, they run entirely on derivatives, the chain rule, and minimization.

Instead of the standard beginner project (classifying handwritten digits from MNIST), this project classifies **music genres from raw audio** — a more interesting problem that produces a real, working demo.

***

## How It Works

A 30-second audio clip goes through the following pipeline:

```
Raw audio (.mp3)
    ↓
librosa extracts 58 numerical features
(MFCCs, tempo, spectral centroid, spectral rolloff, RMS energy, chroma, zero crossing rate)
    ↓
Input vector of 58 numbers fed into the neural network
    ↓
Dense(256, ReLU) → Dense(128, ReLU) → Dense(64, ReLU)
    ↓
Output(10, Softmax) → probability distribution over 10 genres
    ↓
Predicted genre + confidence %
```

The network never "hears" the audio directly — it only sees the 58 features librosa extracts from it.

***

## The Calculus Inside It

This is the part that makes it a Calculus BC project. Training a neural network is, at its core, an applied calculus problem.

**Gradient Descent** — The network learns by minimizing a loss function L that measures prediction error. Weights are updated each epoch using:

$$w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial L}{\partial w}$$

The derivative tells us which direction increases the loss, so we subtract it to move downhill — exactly like using f′(x) to find a minimum.

**Backpropagation and the Chain Rule** — To calculate how a weight in layer 1 affects the loss at the output, the network traces through every layer using the chain rule:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a_4} \cdot \frac{\partial a_4}{\partial a_3} \cdot \frac{\partial a_3}{\partial a_2} \cdot \frac{\partial a_2}{\partial w_1}$$

Without the chain rule, deep learning would not work.

**Sigmoid Derivative** — For the sigmoid activation function σ(x) = 1 / (1 + e^(−x)), the derivative derived using the quotient rule is:

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Loss Regression** — The training loss over 100 epochs was modeled using exponential regression:

$$L(t) = 1.6870 \cdot e^{-0.1078t} + 0.0938$$

Its derivative L′(t) = −0.1818e^(−0.1078t) gives the rate of improvement at any epoch. At epoch 10, the model was improving at −0.0619 loss/epoch. As t → ∞, loss approaches 0.0938 — the floor this architecture cannot improve beyond without changes.

***

## Results

| Metric | Value |
|--------|-------|
| Training accuracy | 98.5% |
| Test accuracy | 75.5% |
| Live demo (Rick Astley — Never Gonna Give You Up) | Predicted: **DISCO**, 76.7% confidence |
| Parameters | 56,906 |
| Epochs | 100 |

The gap between training and test accuracy (98.5% vs 75.5%) reflects overfitting — the model learned the training data well but doesn't generalize perfectly to unseen songs. This is a known limitation of the GTZAN dataset and the network's size.

***

## Dataset

**GTZAN Genre Collection** — 1,000 audio clips, 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock), 100 songs each, 30 seconds per clip.

https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

The features CSV (`features_30_sec.csv`) is included in the dataset download and is what gets loaded into the notebook — the raw audio files are too large to store here.

***

## Run It

1. Open `demo.ipynb` in [Google Colab](https://colab.research.google.com)
2. Mount your Google Drive
3. Place these files in `MyDrive/2026APCalcBCproject/`:
   - `features_30_sec.csv` (from GTZAN on Kaggle)
   - `genre_model.keras`
   - `scaler.pkl`
   - `le.pkl`
4. Run all cells top to bottom

The demo cell downloads a YouTube song, extracts its features, runs the forward pass through each layer visually, and outputs the predicted genre with a softmax probability chart.

***

## Files

| File | Description |
|------|-------------|
| `demo.ipynb` | Full demo: waveform → feature extraction → forward pass → genre prediction |
| `genre_model.keras` | Trained model weights (56,906 params) |
| `report.pdf` | Written report with full calculus derivations |
| `assets/` | Charts: loss regression, ReLU activations, softmax output, sigmoid + derivative, waveform, feature vector |

***

## Requirements

```
tensorflow
librosa
numpy
scikit-learn
scipy
matplotlib
yt-dlp
```

Install in Colab: `!pip install librosa yt-dlp -q` (tensorflow, numpy, sklearn, scipy, matplotlib are pre-installed)

***

*Built by Raj Mankar — AP Calculus BC, 2025–26*
