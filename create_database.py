from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
import os, re

DATA_PATH = "text_data"

print("Current working dir:", os.getcwd())
print("Contents:", os.listdir(DATA_PATH))

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt",loader_cls=TextLoader) #"*.txt" to look up any files in .txt format
    documents = loader.load()
    return documents

documents = load_documents()

def split_by_chapters(text):
    # Split on: *****\nChapter title\n*****
    raw_text = documents[0].page_content

    chapters = re.split(r"\n\*{5}\n", raw_text) # This has the issue of identifying some chapter titles as chapters when a chapter has a double title

    chapter_docs = [
        Document(page_content=c.strip(), metadata={"chapter": i})
        for i, c in enumerate(chapters) if c.strip()
    ]

    chapter_docs = [x for x in chapter_docs if len(x.page_content)>50] # Remove chapter titles 

    return chapter_docs

chapter_docs = split_by_chapters(documents)

#print(len(chapter_docs))
#print(chapter_docs[1].page_content)
#print(len(chapter_docs[4].page_content))

def split_by_dialogue(chapters):
    all_dialogue_docs = []

    for i, chapter in enumerate(chapters):

        raw_text = chapter.page_content

        parts = re.split(r"(?:^|\n)([A-Z]+:\s*)", raw_text) # Using the () means that the speaker name is kept

        #print(parts[0])

        dialogues = []

        if parts[0].strip():
            dialogues.append(parts[0].strip())

        for k in range(1, len(parts), 2):
            dialogues.append(parts[k] + parts[k+1])

        dialogue_docs = [
            Document(page_content=d.strip(), metadata = {"chapter": i, "dialogue": j})
            for j, d in enumerate(dialogues) if d.strip()
        ]

        all_dialogue_docs.extend(dialogue_docs)

    return all_dialogue_docs

all_dialogue_docs = split_by_dialogue(chapter_docs)

print(len(all_dialogue_docs))
print(all_dialogue_docs[0].page_content)
print(all_dialogue_docs[0].metadata)

print ("---------GROUPED INFO----------")

def group_texts_by_chapter(dialogue_docs, group_size=5, overlap=2):
    grouped_docs = []

    # Get unique chapter numbers
    chapters = sorted(set(doc.metadata["chapter"] for doc in dialogue_docs))

    for chapter in chapters:
        # All items in this chapter (dialogue or narration)
        chapter_texts = [doc for doc in dialogue_docs if doc.metadata["chapter"] == chapter]

        i = 0
        while i < len(chapter_texts):
            group_text = "\n\n".join([doc.page_content for doc in chapter_texts[i:i+group_size]])
            dialogue_ids = [doc.metadata["dialogue"] for doc in chapter_texts[i:i+group_size]]
            group_metadata = {
                "chapter": chapter,
                "group_start": min(dialogue_ids),
                "group_end": max(dialogue_ids)
            }
            grouped_docs.append(Document(page_content=group_text, metadata=group_metadata))
            i += group_size - overlap  # Move forward with overlap

    return grouped_docs

grouped_dialogues = group_texts_by_chapter(all_dialogue_docs)

print(len(grouped_dialogues))
print(grouped_dialogues[7].page_content)
print(grouped_dialogues[7].metadata)

#------------ Token Splitting-------------------------

import tiktoken

def tiktoken_split_documents(
    docs,
    model="gpt-5-mini",
    chunk_size=600,
    overlap=150
):
    """
    Splits LangChain Documents using tiktoken.
    """
    encoding = tiktoken.encoding_for_model(model)

    split_docs = []

    for doc in docs:
        tokens = encoding.encode(doc.page_content)

        start = 0
        chunk_id = 0

        while start < len(tokens):
            end = start + chunk_size

            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)

            new_metadata = doc.metadata.copy()
            new_metadata["chunk"] = chunk_id

            split_docs.append(
                Document(
                    page_content=chunk_text,
                    metadata=new_metadata
                )
            )

            start += chunk_size - overlap
            chunk_id += 1

    return split_docs

token_chunks = tiktoken_split_documents(
    grouped_dialogues,
    model="gpt-5-mini",
    chunk_size=1000,
    overlap=350
)

#Check total number of chunks

encoding = tiktoken.encoding_for_model("gpt-5-mini")
total_tokens = sum(len(encoding.encode(doc.page_content)) for doc in token_chunks) # Total of arround 123k tokens --- context window 400k
print(f"Total tokens across all chunks: {total_tokens}")

# Get max tokens in any chunk
max_tokens_in_chunk = max(len(encoding.encode(doc.page_content)) for doc in token_chunks)
print(f"Maximum tokens in a single chunk: {max_tokens_in_chunk}")

### DEBUGGING ###
def debug_print_chunks(chunks, max_print=5):
    """
    Prints info for a few tokenized document chunks for inspection.
    
    chunks: list of LangChain Documents
    max_print: how many chunks to print
    """
    for i, doc in enumerate(chunks[:max_print]):
        tokens = encoding.encode(doc.page_content)
        print(f"----- Chunk {i} -----")
        print(f"Metadata: {doc.metadata}")
        print(f"Token count: {len(tokens)}")
        print(f"Text preview: {doc.page_content}")  # first 500 chars
        print("----------------------\n")

debug_print_chunks(token_chunks, max_print=10)

#--------------- Create the embedding and Load the chromaDB database------------------------

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import openai 
import shutil

CHROMA_PATH = "chroma"

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(model="text-embedding-3-large"), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

save_to_chroma(token_chunks)

