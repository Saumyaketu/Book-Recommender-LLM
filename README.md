# LLM-Powered Book Recommender System

A semantic book recommendation engine that uses Large Language Models (LLMs) and vector search to find books based on natural language queries, specific categories, and emotional tone.

---

## Overview

This project goes beyond simple keyword matching. By leveraging **LangChain**, **ChromaDB**, and **Hugging Face embeddings**, it allows users to describe what they want to read in plain English (e.g., *"a story about a robot discovering humanity"*). The system analyzes the semantic meaning of the query and returns the most relevant books.

Additionally, the system includes an **emotion analysis** feature, allowing users to sort recommendations based on the "tone" of the book (e.g., Happy, Suspenseful, Sad).

---

## Features

* **Semantic Search**: Search for books using natural language descriptions rather than just titles or authors.
* **Emotion Filtering**: Sort recommendations by emotional tone (Happy, Surprise, Suspense, Sad, Fear) using pre-computed sentiment analysis scores.
* **Category Filtering**: Narrow down results by specific genres (Fiction, Nonfiction, Fantasy, etc.).
* **Interactive Dashboard**: A user-friendly web interface built with **Gradio** for easy interaction.
* **Zero-Shot Classification**: Uses LLMs to categorize books that were missing genre tags in the original dataset.

---

## Interface Demo

Here is the dashboard in action:

![Dashboard](screenshots/img1.png)
![Search Result](screenshots/img2.png)
![Search Result](screenshots/img3.png)
![Search Result](screenshots/img4.png)
![Search Result](screenshots/img5.png)

---

## Tech Stack

* **Python**: Core programming language.
* **LangChain**: For managing the retrieval and LLM chains.
* **ChromaDB**: Vector store for efficient semantic search.
* **Hugging Face Transformers**: For text embeddings (`all-MiniLM-L6-v2`) and zero-shot classification (`facebook/bart-large-mnli`).
* **Gradio**: For building the web-based user interface.
* **Pandas & NumPy**: For data manipulation and analysis.

---

## Project Structure

* `gradio_dashboard.py`: The main entry point. Runs the Gradio web application.
* `data/`: Contains the dataset files (`books_cleaned.csv`, `books_with_emotions.csv`, `tagged_description.txt`).
* `vector_search.ipynb`: Notebook demonstrating how the vector database is built and queried.
* `text_classification.ipynb`: Notebook showing how missing categories were filled using zero-shot classification.
* `sentiment_analysis.ipynb`: Notebook used to analyze the emotional tone of book descriptions.
* `data_exploration.ipynb`: Initial data cleaning and exploration.
* `requirements.txt`: List of Python dependencies.

---

## Getting Started

Follow these steps to clone the repository and run the application on your local machine.

### Prerequisites

* Python 3.10 

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/saumyaketu/book-recommender-llm.git](https://github.com/saumyaketu/book-recommender-llm.git)
    cd book-recommender-llm
    ```

2.  **Create a virtual environment:**
    ```bash
    py -3.10 -m venv llm_env
    .\llm_env\Scripts\Activate.ps1
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install langchain-huggingface
    ```

### Running the App

1.  **Launch the dashboard:**
    ```bash
    python gradio_dashboard.py
    ```

2.  **Access the interface:**
    The terminal will output a local URL (usually `http://127.0.0.1:7860`). Open this link in your web browser.

3.  **Use the recommender:**
    * **Query**: Type a description of the book you are looking for.
    * **Category**: (Optional) Select a specific genre from the dropdown.
    * **Tone**: (Optional) Select an emotional tone to prioritize.
    * Click **'Find recommendations'** to see your recommendations!

---

## Data Pipeline

The project follows a structured data pipeline:
1.  **Data Cleaning**: Raw book data is cleaned and processed.
2.  **Text Classification**: Missing categories are predicted using a Zero-Shot Classification model.
3.  **Sentiment Analysis**: Book descriptions are analyzed to determine their emotional probabilities (Joy, Sadness, etc.).
4.  **Vector Embedding**: Book descriptions are converted into vector embeddings using the `all-MiniLM-L6-v2` model and stored in ChromaDB.
5.  **Retrieval**: The dashboard queries ChromaDB to find semantically similar books and filters/sorts them based on user preferences.

---

## References

The project makes use of the following datasets, models, and libraries:

* **Dataset**: [7k Books Dataset](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)
* **Embeddings Model**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Used for creating vector representations of book descriptions.
* **Emotion Classification Model**: [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) - Used for analyzing the emotional tone of the books.
* **Zero-Shot Classification Model**: [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) - Used for predicting missing categories/genres.
* **Libraries**:
    * LangChain
    * ChromaDB
    * Gradio
    * Hugging Face Transformers


