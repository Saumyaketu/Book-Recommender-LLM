import pandas as pd
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")

books = pd.read_csv("data/books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover_not_found.png",
    books["large_thumbnail"]
)

hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db_books = Chroma(
    persist_directory="./chroma_db",
    embedding_function=hf_embeddings
)

def generate_explanation(query: str, recommendations: str) -> str:
    prompt = f"User asked for: '{query}'. Based on these books: {recommendations}, explain in 2-3 sentences why these were chosen."
    response = llm.invoke(prompt)
    return response.content

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 500,
        final_top_k: int = 16,
) -> pd.DataFrame:
    
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [str(rec.page_content.strip('"').split()[0]).strip() for rec in recs]
    books["isbn13_str"] = books["isbn13"].astype(str).str.split('.').str[0].str.strip()

    book_recs = books[books["isbn13_str"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprise":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(query: str, category: str, tone: str, history: list):
    if history is None:
        history = []

    recommendations = retrieve_semantic_recommendations(query, category, tone)
    book_titles = ", ".join(recommendations["title"].dropna().tolist())
    
    explanation = generate_explanation(query, book_titles)
    
    results = []
    for _, row in recommendations.iterrows():
        description = row["description"]
        if pd.isna(description):
            truncated_description = "No description available."
        else:
            truncated_desc_split = str(description).split()
            truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors = row["authors"]
        if pd.isna(authors):
            authors_str = "Unknown Author"
        else:
            authors_split = str(authors).split(";")
            if len(authors_split) == 2:
                authors_str = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            else:
                authors_str = str(authors)

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
        
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": explanation})
    
    return results, history, ""

def get_categories():
    return ["All"] + sorted(books["simple_categories"].unique())

def get_tones():
    return ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]