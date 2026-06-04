# music-genre-classifier
Final Project for my AP Calculus BC class

A 4-layer neural network that classifies music into 10 genres from raw audio.
Built for AP Calculus BC (2025–26) to connect neural network math to calculus.

**75% test accuracy** on 1,000 songs across blues, classical, country, disco,
hip-hop, jazz, metal, pop, reggae, and rock.

## How it works
Raw audio → librosa extracts 58 features (MFCCs, tempo, spectral centroid, etc.)
→ feedforward neural net → softmax probabilities → predicted genre

## Architecture
    Input(58) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(64, ReLU) → Output(10, Softmax)

## Calculus connections
- Gradient descent: w = w − lr × dL/dw
- Backpropagation: chain rule across 4 layers
- Sigmoid derivative: quotient rule → σ'(x) = σ(x)(1 − σ(x))
- Loss curve fit: L(t) = 1.687·e^(−0.108t) + 0.094

## Dataset
GTZAN Genre Collection — 1,000 audio clips, 10 genres
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

## Run it
Open demo.ipynb in Google Colab and run all cells.
Model files (genre_model.keras, scaler.pkl, le.pkl) must be in your Drive
at MyDrive/2026APCalcBCproject/

## Files
- demo.ipynb — full walkthrough: waveform → features → forward pass → prediction
- genre_model.keras — trained model (56,906 params)
- report.pdf — written report with full calculus derivations
- assets/ — charts: loss curve, activations, softmax output, sigmoid plots

## Requirements
    tensorflow  librosa  numpy  scikit-learn  scipy  matplotlib  yt-dlp
