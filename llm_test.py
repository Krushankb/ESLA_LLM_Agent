from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import ollama
import warnings

warnings.filterwarnings("ignore")

loaders = [
    # TextLoader("./temp.txt"),
    TextLoader("./lesson_plans/plan2.txt")
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

def ollama_llm_open(user_input, context, history):
    conversation = ""
    for turn in history:
        conversation += f"User: {turn['user_input']}\nAssistant: {turn['llm_response']}\n"
    # conversation += f"User: {question}\n"

    prompt = (
        """
        You are a friendly and supportive English learning assistant for International Teaching Assistants. Assume the user you are speaking with is an International Teaching Assistant (ITA) at a US postsecondary institution. They are learning English and are not native speakers. They are preparing to teach a class and need to practice their English skills.
        
        You are to help them practice conversations using academic language and help them to correctly use English when they are asking questions.
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
        f"Current User Input: {user_input}\n\n"
    )
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def ollama_llm_generate_scenario(context):
    prompt = (
       f"""
        You are a friendly and supportive English learning assistant for International Teaching Assistants. 

        Generate a scenario that an International Teaching Assistant might encounter in a US postsecondary institution. The scenario should be realistic and relevant to the academic context.
        
        Possible scenarios include:
        - Explaining a topic to students during a lecture
        - Responding to a student's question in class
        - Handling a difficult situation with a student

        The scenario you output should be a short paragraph that describes a situation where a student requests help in some form to the user, who will act as the International TA. For instance:

        A student approaches you after class and asks for clarification on the topic of Big-O notation. They say they are confused about the difference between O(n) and O(n^2) time complexity. Please explain the difference in a way that is easy for them to understand.
        
        Make sure to address the user as "you" in the scenario, where the user is the ITA.

        The scenario should be based on the context provided.

        Context:
        {context}
        """
    )
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def ollama_llm_scenario_eval(scenario, user_input):
    prompt = (
        f"""
        You are evaluating an international TA's response to an academic scenario. The goal of this evaluation is to help the ITA's abilities in handling a real-world academic scenario using English. Your task is to provide feedback on their response, focusing on the following aspects:
        1. Clarity and conciseness of the response
        2. Use of proper English grammar and vocabulary
        3. Use of proper academic language and style for a US postsecondary institution
        4. Accuracy of the response given the scenario

        This is the scenario the user is responding to:
        {scenario}

        Please evaluate this response given by the user:
        {user_input}

        Evaluate the response by addressing the user directly as "you". Highlight areas they did well in, and provide suggestions for improvement. If there are any English errors, list them and provide corrections.
        """
    )
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def rag_chain_open(user_input, history):
    retrieved_docs = retriever.invoke(user_input)
    formatted_context = combine_docs(retrieved_docs)
    llm_response = ollama_llm_open(user_input, formatted_context, history)
    history.append({"user_input": user_input, "llm_response": llm_response})
    return llm_response

def rag_chain_generate_scenario(user_input):
    retrieved_docs = retriever.invoke(user_input)
    formatted_context = combine_docs(retrieved_docs)
    scenario = ollama_llm_generate_scenario(formatted_context)
    return scenario

def rag_chain_eval_scenario(scenario, user_response, history):
    eval = ollama_llm_scenario_eval(scenario, user_input)
    history.append({"scenario": scenario, "user_response": user_response, "llm_eval": eval})
    return eval

history = []

print("Welcome to ESL Agent for ITAs. You can have free conversation with the LLM to practice your English or ask any questions.\n")
print("You can also type 'scenario' to generate a scenario for you to respond to. The agent will then evaluate your response to the scenario.\n")
print("You can type 'exit' to end the session.\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("Goodbye!")
        break
    if user_input.strip() == "scenario":
        print("Generating scenario...\n")
        scenario = rag_chain_generate_scenario(user_input)
        print(f"Scenario: {scenario}\n")
        user_response = input("Your response to the scenario: ")
        if user_response.strip().lower() == "exit":
            print("Goodbye!")
            break
        print("Evaluating your response...\n")
        eval = rag_chain_eval_scenario(scenario, user_response, history)
        print(f"Assistant: \n\n {eval}\n")
    else:
        response = rag_chain_open(user_input, history)
        print(f"Assistant: {response}\n")
