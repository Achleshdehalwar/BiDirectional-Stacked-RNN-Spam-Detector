# 🛡️ Bi-Directional Stacked RNN Spam Detector

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Optuna](https://img.shields.io/badge/Optuna-4BB4E6?style=for-the-badge&logo=python&logoColor=white)](https://optuna.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue?style=for-the-badge)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **A robust End-to-End NLP pipeline engineering a custom Stacked Bi-Directional RNN to detect SMS spam with high precision.**

🔗 **[Live Demo on Hugging Face](https://huggingface.co/spaces/achleszh/BiDirectional-Stacked-RNN-Spam-Detector)**

---

## 📖 Overview

This project is not just a standard implementation of a neural network; it is an engineering solution to the specific challenges of text classification using Recurrent Neural Networks (RNNs).

Most vanilla RNNs suffer from the **Vanishing Gradient Problem**, where the model "forgets" early information in a sequence. To solve this without resorting to heavier Transformers, I engineered a **Bi-Directional Stacked** architecture that:
1.  **Reads Context:** Processes text from both Left-to-Right and Right-to-Left.
2.  **Optimizes Depth:** Uses stacked hidden layers to learn hierarchical patterns.
3.  **Ensures Stability:** Implements strict sequence windowing (`MAX_SEQ_LEN=20`) to maintain gradient flow.

---

## ⚙️ Architecture

The core model is a custom PyTorch module designed for stability and performance:

* **Embedding Layer:** Maps sparse one-hot indices to dense vectors.
* **Bi-Directional RNN Layers:**
    * *Layer 1 (Stacked):* Captures low-level features, returning full sequences.
    * *Layer 2 (Head):* Captures high-level abstractions, returning the final hidden state.
* **Feature Fusion:** Concatenates the final Forward and Backward hidden states.
* **Classifier:** A fully connected layer with Sigmoid activation.

---

## 🛠️ Tech Stack

* **Deep Learning Framework:** `PyTorch` (Custom Model Definition)
* **Hyperparameter Tuning:** `Optuna` (Automated Bayesian Optimization)
* **Data Processing:** `Pandas` & `NumPy`
* **Frontend/Deployment:** `Streamlit`
* **Dataset:** UCI SMS Spam Collection (via `kagglehub`)

---

## 🚀 Installation & Usage

### 1. Clone the Repository
`bash
git clone [https://github.com/yourusername/rnn-spam-detector.git](https://github.com/yourusername/rnn-spam-detector.git)
cd rnn-spam-detector`

### 2. Install Dependencies
  `pip install -r req.txt`

### 3. Train the Model (Local)
Run the training script to fetch data, run Optuna optimization, and generate the model artifacts (.pth, .json).
`python train_and_save.py`
Output: Generates spam_rnn.pth, vocab.json, and config.json.

### 4. Run the App
Launch the Streamlit interface locally.
`streamlit run app.py`

### 📂 Project Structure
├── app.py                 # The Streamlit application (Inference Engine)
├── train_and_save.py      # Training pipeline + Optuna Optimization
├── requirements.txt       # Project dependencies
├── spam_rnn.pth           # Saved PyTorch Model Weights (Generated)
├── vocab.json             # Learned Vocabulary Dictionary (Generated)
├── config.json            # Model Architecture Config (Generated)
└── README.md              # Documentation

### 📊 Optimization (Optuna)
Instead of guessing hyperparameters, this project integrates Optuna to mathematically find the best configuration.
Search Space:
Embedding Dimension: [32 - 128]
Hidden Dimension: [32 - 128]
Dropout: [0.1 - 0.5]
Learning Rate: [1e-4 - 1e-2] (Log Scale)
The training script automatically prunes unpromising trials to save compute resources.

### 🤝 Contributing
1. Contributions, issues, and feature requests are welcome!
2. Fork the Project
3. Create your Feature Branch (git checkout -b feature/AmazingFeature)
4. Commit your Changes (git commit -m 'Add some AmazingFeature')
5. Push to the Branch (git push origin feature/AmazingFeature)
6. Open a Pull Request

### 📝 License
Distributed under the MIT License. See LICENSE for more information.
<p align="center">
Built with ❤️ using PyTorch & Streamlit
</p>
