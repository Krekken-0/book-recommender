import pandas as pd
import numpy as np

import chromadb

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

import gradio as gr

books = pd.read_csv("books_with_emotion.csv")
books["large_thumbnail"] = books['thumbnail'] + "&fife=w800"
books["large_thumbnail"] = np.where(books["large_thumbnail"].isna(), "cover-not-found.jpg", books["large_thumbnail"])

# raw_documents = TextLoader("tagged_descriptions.txt").load()
# text_splitter = CharacterTextSplitter(separator="\n", chunch_size=0.01, chunk_overlap=0)
# documents = text_splitter.split_documents(raw_documents)
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db_new")
db_books_new = Chroma(
    client=client,
    embedding_function=embeddings_model
)

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:
    recs = db_books_new.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        books_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        books_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        books_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        books_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        books_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        books_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        books_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return books_recs

def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desp_split = description.split()
        truncated_description = " ".join(truncated_desp_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{','.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

categories = ["All"] + ["Children's Fiction", "Children's Nonfiction", "Fiction", "Nonfiction" ]
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g. A story about forgiveness")
    with gr.Row():
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category",
                                    value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone",
                                value = "All")

    submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = None, rows = 4, container=True, object_fit='scale-down',
                        show_fullscreen_button=False,)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)

if __name__ == "__main__":
    dashboard.launch()




