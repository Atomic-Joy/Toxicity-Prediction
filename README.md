# Multi-Task Toxicity Prediction with Graph Attention Networks

A comprehensive implementation of multi-task toxicity prediction using Graph Attention Networks (GATs) on the Tox21 dataset. This project demonstrates advanced machine learning techniques for computational toxicology and drug discovery.

## 🎯 Overview

This repository implements a state-of-the-art Graph Attention Network for predicting chemical toxicity across multiple endpoints simultaneously. The model learns molecular representations from SMILES strings and predicts toxicity profiles for 12 different assays, enabling efficient screening of chemical compounds for drug discovery and safety assessment.

## ✨ Key Features

### 🧪 Molecular Graph Processing
- **SMILES to Graph Conversion**: Transform chemical structures into graph representations
- **Rich Atomic Features**: Atomic number, degree, formal charge, hybridization, aromaticity
- **PyTorch Geometric Integration**: Efficient graph neural network processing

### 🧠 Advanced Neural Architecture
- **Multi-Head Graph Attention**: Captures complex molecular interactions
- **Multi-Task Learning**: Simultaneous prediction across 12 toxicity endpoints
- **Attention Interpretability**: Extract and visualize attention weights

### 📊 Comprehensive Evaluation
- **ROC-AUC Metrics**: Performance evaluation for each toxicity task
- **Calibration Analysis**: Assess prediction confidence and reliability
- **Confusion Matrices**: Detailed classification performance breakdown

### 🔍 Model Interpretability
- **Attention Maps**: Visualize important molecular substructures
- **Feature Importance**: Understand model decision-making
- **Interactive Visualizations**: Molecular structure overlays

## 🏗️ Architecture

### Model Components

```python
class ToxicityGAT(nn.Module):
    def __init__(self, num_features, num_outputs):
        super().__init__()
        self.gat1 = GATConv(num_features, 64, heads=4)  # Multi-head attention
        self.gat2 = GATConv(64*4, 128, heads=1)         # Single-head attention
        self.dropout = nn.Dropout(0.3)                   # Regularization
        self.fc = nn.Linear(128, num_outputs)            # Multi-task output
```

### Training Pipeline

1. **Data Loading**: Load Tox21 dataset with SMILES and toxicity labels
2. **Graph Construction**: Convert molecules to graph data structures
3. **Model Training**: 100 epochs with Adam optimizer and masked loss
4. **Evaluation**: Comprehensive metrics and visualization analysis

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Dependencies

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install rdkit
pip install scikit-learn matplotlib seaborn pandas numpy
```

### Clone Repository

```bash
git clone https://github.com/Atomic-Joy/Toxicity-Prediction.git
cd toxicity-prediction-gat
```

## 📖 Usage

### Quick Start

1. **Jupyter Notebook** (Recommended for exploration):
   ```bash
   jupyter notebook Toxicity_MT.ipynb
   ```

2. **Python Script**:
   ```bash
   python Toxicity_MT.py
   ```

### Model Inference

```python
from Toxicity_MT import ToxicityGAT
import torch

# Load trained model
checkpoint = torch.load("best_model.pth")
model = ToxicityGAT(checkpoint["num_features"], checkpoint["num_outputs"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Make predictions on new molecules
# (See notebook for complete inference pipeline)
```

## 📊 Dataset

### Tox21 Dataset

- **Source**: NIH Tox21 initiative
- **Size**: ~8,000 chemical compounds
- **Tasks**: 12 toxicity endpoints (nuclear receptors, stress response, etc.)
- **Format**: SMILES strings with binary toxicity labels
- **Challenge**: Missing labels, class imbalance, multi-task nature

### Data Statistics

- **Molecular Size**: 10-30 atoms per molecule
- **Class Balance**: Varies significantly across tasks
- **Missing Data**: ~20-30% missing labels per task

### Key Achievements

- **Multi-Task Performance**: Consistent performance across diverse toxicity endpoints
- **Interpretability**: Attention mechanisms reveal important molecular substructures
- **Scalability**: Efficient processing of molecular graphs of varying sizes
- **Robustness**: Handles missing data and class imbalance effectively

## 📊 Visualization

The repository includes comprehensive visualization capabilities:

- **Molecular Structures**: 2D chemical structure rendering
- **Attention Maps**: Heatmaps showing important atoms for toxicity
- **Training Curves**: Loss and AUC progression during training
- **Performance Analysis**: ROC curves, confusion matrices, calibration plots
- **Dataset Insights**: Correlation matrices, class balance distributions

### Example Visualizations

- Training progress with loss and AUC curves
- Attention-weighted molecular structures
- Task performance heatmaps
- ROC curves for all toxicity endpoints
---
