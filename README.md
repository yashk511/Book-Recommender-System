# 📚 Book Recommendation System

> A practical, end‑to‑end book recommender that combines data cleaning, sentiment analysis, text classification, and vector similarity search with a simple Gradio UI.

---

## 🔎 Overview

This project builds a content‑based **Book Recommendation System**. It cleans and enriches book metadata, classifies descriptions, extracts sentiment from reviews, and indexes everything into a vector store for **semantic search** and **recommendations**. A lightweight **Gradio** app lets you try queries like *“books about mental-wellbeing”* or *“books to teach children about nature.”*

---

## ✨ Features

* **Semantic search** with sentence embeddings + **ChromaDB** vector store
* **Text classification** of book descriptions for better filtering
* **Sentiment & emotion tagging** for reviews to improve ranking
* **Cleaned dataset** with unified fields and tags
* **Gradio UI** for interactive querying and quick demos

---

**sample query:**

* *“wholesome family read with adventure”*

![Demo UI](ui_screenshot1.png)
![Demo UI](ui_screenshot2.png)

---

## 🧰 Tech Stack

* **Python**: pandas, numpy, seaborn
* **NLP/Embeddings**: transformers, sentence‑transformers (Hugging face)
* **Vector Store**: chromadb
* **Orchestration/Utils**: langchain , tqdm, regex
* **UI**: gradio
* **Notebooks**: Jupyter

---

## 🚀 Usage Instructions

1. **Install dependencies for the Gradio app:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: `requirements.txt` only includes packages needed to run `app.py` (the Gradio dashboard).*

2. **If you want to run the Jupyter notebooks for data cleaning, classification, or sentiment analysis, you may need to install additional packages:**
   ```bash
   pip install transformers tqdm seaborn matplotlib
   ```

3. **Run each notebook in order for data preparation and model building (optional).**

4. **Launch the dashboard:**
   ```bash
   python app.py
   ```

5. **Access the dashboard at** `http://127.0.0.1:7860` **in your browser.**

---

## 📄 License

MIT License — see `LICENSE` for details.

---
