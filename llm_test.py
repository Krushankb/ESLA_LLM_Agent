from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import ollama
import warnings

warnings.filterwarnings("ignore")

loaders = [
    TextLoader("./temp.txt"),
    TextLoader("./sample_lesson_plans.txt")
]

print("Loading documents...\n")

docs = []
for loader in loaders:
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096)
splits = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="llama3")
print("Creating vectorstore...\n")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever()
print("Retriever is ready!\n")

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ollama_llm(question, context, history):
    conversation = ""
    for turn in history:
        conversation += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"
    conversation += f"User: {question}\n"

    prompt = (
        "Use the documents and the conversation to answer the user's question. Make sure to read everything carefully to not miss anything."
        "Be helpful, relevant, and concise.\n\n"
        f"Context:\n{context}\n\n"
        f"Conversation History:\n{conversation}"
    )
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def rag_chain(question, history):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    answer = ollama_llm(question, formatted_context, history)
    history.append({"question": question, "answer": answer})
    return answer

history = []

print("Chat initialized. Type 'exit' to end the conversation.\n")
while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("Goodbye!")
        break
    response = rag_chain(user_input, history)
    print(f"Assistant: {response}\n")