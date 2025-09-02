import argparse
import os
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from embedding_function import embedding_function

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Ensure the Chroma directory exists
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"Database not found at {CHROMA_PATH}. Please build the database first using the Database tab in the GUI or run 'python database.py'")
    
    # Prepare the DB.
    embeddings = embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=api_key)
    response_text = llm.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text.content}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()