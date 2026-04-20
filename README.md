
# 🧠 Self-Pruning Neural Network

🚀 A neural network that **learns to prune its own weights during training**, reducing model size while maintaining performance—no separate pruning step required.

---

## 📌 Overview

Deep neural networks are often **over-parameterized**, leading to inefficiency.
This project introduces a **self-pruning mechanism** where each weight is controlled by a learnable gate that determines its importance.

* Important weights are retained
* Unimportant weights are suppressed
* Model becomes **compact automatically**

---

## 🧠 Core Idea

Each weight is scaled by a learnable gate:

```python
gate = sigmoid(gate_score)
weight = weight * gate
```

### Loss Function

```python
loss = classification_loss + λ * L1(gates)
```

* Gate → **0** ⇒ pruned
* Gate → **1** ⇒ active

---

## 🏗️ Architecture

```
Input (CIFAR-10)
   ↓
Flatten
   ↓
FC (1024) → BN → ReLU
   ↓
FC (512)  → BN → ReLU
   ↓
FC (256)  → BN → ReLU
   ↓
FC (10)
```

All layers use a custom **PrunableLinear** module.

---

## 📊 Results

### 🔹 Gate Distribution

![Gate Distribution](results/plots.png)


---

## 📁 Project Structure

```
model.py    # PrunableLinear + model
train.py    # Training loop
utils.py    # Data + evaluation
plots.py    # Visualizations
results/    # Outputs (auto-generated)
```

---

## ⚙️ Tech Stack

* PyTorch
* NumPy
* Matplotlib

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python train.py
```

Run experiments:

```bash
python train.py --lambdas 0.0001 0.001 0.01 --epochs 60
```

---

## 🔍 Key Highlights

* ✅ Dynamic pruning during training
* ✅ No post-processing required
* ✅ Learnable sparsity via L1 regularization
* ✅ Accuracy vs efficiency trade-off


---

## 🔮 Future Work

* CNN-based architecture
* Structured pruning (filters/channels)
* Advanced sparsity methods (L0 / Hard Concrete)
