from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma

import gradio as gr
import pandas as pd
import numpy as np

books = pd.read_csv('books_with_emotions.csv')

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isnull(),
    "Book_cover_not_found.png",
    books["large_thumbnail"],
)

# Load the file (force UTF-8 to avoid UnicodeDecodeError)
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()

# Use a large chunk size instead of 0 (0 is not valid now)
text_splitter = CharacterTextSplitter(
    chunk_size=10_000,  # large enough so text isn't split
    chunk_overlap=0,
    separator="\n"
)

documents = text_splitter.split_documents(raw_documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

# 2. Reranker for better sorting (optional but recommended)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# 3. Build/Load vector store
db_books = Chroma.from_documents(
    documents, 
    embedding_model
)

def retrieve_semantic_recommendations(
        query: str,
        category: str = "ALL",
        tone: str = "All",
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:
    # Perform semantic search
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    
    # Unpack tuples (Document, score)
    books_list = [int(doc.page_content.strip('"').split()[0]) for doc, _ in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    # Category filtering
    if category != "ALL":
        book_recs = book_recs[book_recs["simple_category"] == category].head(final_top_k)

    # Tone sorting
    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs


def recommend_books(
        query: str,
        category: str = "None",
        tone: str = "None",
): 

    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

categories = ["ALL"] + sorted(books["simple_category"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="ALL")
tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "ALL")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)
    

if __name__ == "__main__":
    dashboard.launch()