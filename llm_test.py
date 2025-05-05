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
        """
        You are a friendly and supportive English learning assistant for International Teaching Assistants. You are to help them practice conversations using academic language and help them to correctly use English when they are asking questions.
        Make sure to do both!

        Please respond in a way that:
        1. Maintains natural conversation flow
        2. Gently corrects any English errors
        3. Uses academic vocabulary when appropriate
        4. Encourages further discussion
        5. Models proper academic speaking style

        Your response must follow this format:

        Response: *Your response to the conversation.*
        Suggestions: *Provide any suggestions on clarity and conciseness for the user's answers.*
        English Corrections: 
        - List any misused or incorrect English statements here, then give a short reason why it's wrong and a correct way of saying it.

        For example, if the user enters "how to explain student complex topic", a sample response is:
        Response: That's a great question about explaining complex topics! When teaching students, it's helpful to break down difficult concepts into smaller, more manageable parts. For example, if you're explaining a mathematical concept, you might start with a simple example before moving to more complex applications.
        English Corrections:
        - Instead of "how to explain student complex topic", you can say "how to explain complex topics to students" for clearer word order and article usage.

        Below, you are given the context in the form of a document, as well as the conversation history to tailor your response.
        """
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
