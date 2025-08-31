📘 Sarcasm Detection in News Headlines (NLP Project)

This project implements sarcasm detection in news headlines using multiple word embeddings and machine learning models.
The goal is to compare different embedding techniques and analyze their performance across various classifiers.

📂 Project Structure
Sarcasm-Detection-NLP/
│── README.md
│── requirements.txt
│── sarcasm_detection.ipynb
│── results/ (graphs, confusion matrix, accuracy plots)

🚀 Features

We experimented with 3 types of embeddings:

Traditional Word Embeddings

Count Vectorizer

TF-IDF Vectorizer

Co-occurrence Matrix

Static Word Embeddings

Word2Vec

GloVe

FastText

Contextualized Word Embeddings

ELMo

GPT-2

BERT

Models Implemented:

Logistic Regression

Support Vector Classifier (SVM)

📊 Results (Sample — update with your results table)

Traditional embeddings: 70–78% accuracy

Static embeddings: 80–85% accuracy

Contextualized embeddings (BERT, GPT-2, ELMo): best performance (85–90% accuracy)

(Replace with your exact accuracy/F1 scores once finalized.)

🛠️ Tech Stack

Languages: Python

Libraries: Pandas, NumPy, NLTK, Scikit-learn, TensorFlow/Keras, Matplotlib, Transformers

⚙️ How to Run

Clone the repository

git clone https://github.com/yourusername/Sarcasm-Detection-NLP.git
cd Sarcasm-Detection-NLP


Install dependencies

pip install -r requirements.txt


Run the notebook

jupyter notebook sarcasm_detection.ipynb

📜 Dataset

News Headlines Dataset for Sarcasm Detection (Kaggle)

🙌 Author

Prateek Sharma

B.Tech CSE (AIML) | Aspiring AI/ML Engineer

LinkedIn
 | GitHub
