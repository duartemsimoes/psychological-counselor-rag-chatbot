import argparse
# from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.schema import HumanMessage

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a calm, thoughtful counsellor.

Use the following dialogue as your main philosophical framework.
Base your answer primarily on the principles in the text
(e.g. separation of tasks, community feeling, horizontal relationships, encouragement),
but you may use your general knowledge to clarify, explain, and apply them practically.

Dialogue context:
{context}

---

Question: {question}

Respond in a clear, connected, and practical manner. Make sure your answer:
- Clearly addresses the question
- Reflects the ideas in the dialogue, linking them naturally into your explanation
- Provides actionable advice that can be applied in real life
"""

FALLBACK_PROMPT = """
You are a calm, thoughtful counsellor inspired by Adlerian psychology.

No relevant passages were found in the provided knowledge base for this question.

Answer the following question ONLY using your general knowledge of
Adlerian psychology (concepts like community feeling, encouragement,
purpose, belonging, and horizontal relationships).

Clearly state at the beginning that you are NOT using retrieved sources.

Question: {question}

Respond in a clear, connected, and practical manner. Make sure your answer:
- Clearly addresses the question
- Reflects the ideas in the dialogue, linking them naturally into your explanation
- Provides actionable advice that can be applied in real life
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Initialize a small, fast LLM for classification
    classifier = ChatOpenAI(temperature=0, model="gpt-5-nano")

    # Ask it if the question is related to psychology
    classification = classifier.invoke([
        HumanMessage(content=f"""
    Is the following question related to psychology, human emotions,
    relationships, or human behavior? Answer ONLY yes or no.

    Question: {query_text}
    """)
    ])

    # Check the answer
    if classification.content.strip().lower() != "yes":
        print("This assistant only answers psychology-related questions.")
        return

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB. Use a threshold to deal with questions that in nothing are related to the purpose of the RAG model.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    # Check the retrieved documents to see if you should use the information from RAG
    use_rag = not (len(results) == 0 or results[0][1] < 0.05)

    
    if use_rag:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
    else:
        prompt_template = ChatPromptTemplate.from_template(FALLBACK_PROMPT)
        prompt = prompt_template.format(question=query_text)
    
    #print(prompt)

    model = ChatOpenAI(model="gpt-5-mini")

    response = model.invoke([
        HumanMessage(content=prompt)
    ])

    response_text = response.content

    if use_rag:
        sources = sorted({doc.metadata.get("chapter") for doc, _score in results})
        formatted_response = f"Response: {response_text}\nSources: Chapters {sources}"
    else:
        formatted_response = (
            "⚠️ No relevant documents were found in the knowledge base.\n\n"
            f"Response (Adlerian-based, no sources used): {response_text}"
        )
    print(formatted_response)

if __name__ == "__main__":
    main()