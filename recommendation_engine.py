import pandas as pd
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from sentence_transformers import CrossEncoder

llm = ChatOllama(model="llama3.2")

# Initialized Cross-Encoder for precision re-ranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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

CURRENT_BOOKS_DF = pd.DataFrame()

# ---------------------------------------------------------
# LANGGRAPH TOOLS
# ---------------------------------------------------------
@tool
def search_chroma_db(query: str, category: str = "All") -> str:
    """
    Searches the database for books matching the user's natural language query.
    Optionally filters by category and re-ranks results with a Cross-Encoder.
    ALWAYS call this tool first to get book recommendations.
    """
    print(f"\n[AGENT ACTION]: Searching database for '{query}' (Category: {category})...")
    
    global CURRENT_BOOKS_DF
    recs = db_books.similarity_search(query, k=500)
    books_list = [str(rec.page_content.strip('"').split()[0]).strip() for rec in recs]
    books["isbn13_str"] = books["isbn13"].astype(str).str.split('.').str[0].str.strip()

    book_recs = books[books["isbn13_str"].isin(books_list)].copy()
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]
        
    candidates = book_recs.head(40).copy()
    
    if not candidates.empty:
        print("[CROSS-ENCODER]: Re-ranking candidates for high relevance...")
        pairs = [[query, f"{row['title']}: {str(row['description'])}"] for _, row in candidates.iterrows()]
        
        scores = reranker.predict(pairs)
        candidates["cross_score"] = scores
        
        candidates.sort_values(by="cross_score", ascending=False, inplace=True)
        
    CURRENT_BOOKS_DF = candidates.head(16).copy()
    
    summary = "Retrieved Books:\n"
    for _, row in CURRENT_BOOKS_DF.iterrows():
        desc = str(row['description']).split()[:15]
        short_desc = " ".join(desc) + "..."
        summary += f"- {row['title']}: {short_desc}\n"
    return summary

@tool
def filter_by_emotion(tone: str) -> str:
    """
    Sorts the currently retrieved books by an emotional tone.
    Allowed tones: 'Happy', 'Surprise', 'Angry', 'Suspenseful', 'Sad'.
    Call this tool AFTER search_chroma_db if the user specifies an emotion.
    """
    print(f"\n[AGENT ACTION]: Filtering current results by emotion: '{tone}'...")
    
    global CURRENT_BOOKS_DF
    if CURRENT_BOOKS_DF.empty:
        return "No books retrieved yet. Call search_chroma_db first."
        
    if tone == "Happy":
        CURRENT_BOOKS_DF.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprise":
        CURRENT_BOOKS_DF.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        CURRENT_BOOKS_DF.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        CURRENT_BOOKS_DF.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        CURRENT_BOOKS_DF.sort_values(by="sadness", ascending=False, inplace=True)
        
    return f"Successfully sorted the current books by {tone}."


# Initialized the LangChain Agent
tools = [search_chroma_db, filter_by_emotion]
agent_executor = create_agent(llm, tools)

# ---------------------------------------------------------
# RECOMMENDATION LOGIC
# ---------------------------------------------------------
def recommend_books(query: str, category: str, tone: str, history: list):
    global CURRENT_BOOKS_DF
    
    if history is None:
        history = []

    system_prompt = (
        "You are an expert book recommender. You have access to tools. "
        "You MUST use 'search_chroma_db' to find books. "
        "If a specific emotion is requested, you MUST use 'filter_by_emotion' afterwards. "
        "Once tools are executed, write a 2-3 sentence explanation of why you chose these specific books."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append(msg)
        
    user_instruction = f"User Request: {query}\nCategory: {category}\nEmotion: {tone}"
    messages.append({"role": "user", "content": user_instruction})
    
    response = agent_executor.invoke({"messages": messages})
    final_answer = response["messages"][-1].content
    
    results = []
    if not CURRENT_BOOKS_DF.empty:
        for _, row in CURRENT_BOOKS_DF.iterrows():
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
    history.append({"role": "assistant", "content": final_answer})
    
    return results, history, ""

def get_categories():
    return ["All"] + sorted(books["simple_categories"].unique())

def get_tones():
    return ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]