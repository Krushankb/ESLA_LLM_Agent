import os
import time
import json
import warnings
import re
from typing import List, Dict, Any, Tuple

import gradio as gr

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import ollama

# -----------------------------------------------------------------------------
# Configuration & Globals
# -----------------------------------------------------------------------------

warnings.filterwarnings("ignore")

# Select the GPU you want Ollama to use (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# In‑memory session‑level state. This will be wrapped by a gr.State component so
# that each browser session gets its own copy.
SessionState = Dict[str, Any]


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def is_scenario_request(text: str) -> bool:
    """Return True if the user's message is asking for a scenario."""
    return bool(re.search(r"\bscenario\b", text.lower()))


def build_retriever(file_path: str):
    """Load a lesson‑plan file and return (retriever, raw_text)."""
    loader = TextLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=4096)
    splits = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = Chroma.from_documents(splits, embedding=embeddings)
    return vectorstore.as_retriever(), "\n".join(page.page_content for page in docs)


# -----------------------------------------------------------------------------
# LLM helper functions (same prompt logic as the original script)
# -----------------------------------------------------------------------------

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def ollama_llm_open(user_input: str, context: str, history: List[dict]):
    conversation = ""
    for turn in history:
        if "user_input" in turn:
            conversation += (
                f"User: {turn['user_input']}\nAssistant: {turn['llm_response']}\n"
            )
        elif "scenario" in turn:
            conversation += (
                f"Scenario: {turn['scenario']}\nUser Response: {turn['user_response']}\n"
                f"Evaluation: {turn['llm_eval']}\n"
            )

    prompt = f"""
You are a friendly and supportive English learning assistant for International Teaching Assistants. Assume the user you are speaking with is an International Teaching Assistant (ITA) at a US postsecondary institution. They are learning English and are not native speakers. They are preparing to teach a class and need to practice their English skills.

You are to help them practice conversations using academic language and help them to correctly use English when they are asking questions.
Make sure to do both!

Please respond in a way that:
1. Maintains natural conversation flow
2. Gently corrects any English errors
3. Uses academic vocabulary when appropriate
4. Encourages further discussion
5. Models proper academic speaking style

You are given the lesson plan as well as the conversation history to tailor your response. The conversation history can also contain past scenarios provided to the user and their responses to those scenarios.

Context: {context}

Conversation History:
{conversation}

Give your response to the following user input keeping in mind the context and conversation history.

Current User Input: {user_input}

Your response must follow this format:

Response: *Your response to the conversation.*
Suggestions (option): *Provide any suggestions on clarity and conciseness for the user's answers.*
English Corrections:
- List any misused or incorrect English statements here, then give a short reason why it's wrong and a correct way of saying it.
"""

    response = ollama.chat(
        model="llama3", messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


def ollama_llm_generate_scenario(context: str):
    prompt = f"""
Generate a scenario that an International Teaching Assistant might encounter in a US postsecondary institution. The scenario should be realistic and relevant to the academic context.

The scenario you output should be a short paragraph that describes a situation where a student requests help in some form to the user, who will act as the International TA. For instance:

A student approaches you after class and asks for clarification on the topic of Big‑O notation. They say they are confused about the difference between O(n) and O(n^2) time complexity. Please explain the difference in a way that is easy for them to understand.

Make sure to address the user as "you" in the scenario, where the user is the International TA. Your scenario should end with a question or request for help from the student to the user, who will respond as a TA.

The scenario should be based on the lesson plan provided.

Lesson Plan:
{context}
"""
    response = ollama.chat(
        model="llama3", messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


def ollama_llm_scenario_eval(scenario: str, user_input: str):
    prompt = f"""
You are evaluating an international TA's response to an academic scenario. The goal of this evaluation is to help the ITA's abilities in handling a real‑world academic scenario using English. Your task is to provide feedback on their response, focusing on the following aspects:
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
    response = ollama.chat(
        model="llama3", messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# -----------------------------------------------------------------------------
# RAG Chain wrapper functions
# -----------------------------------------------------------------------------

def rag_chain_open(user_input: str, state: SessionState):
    retrieved_docs = state["retriever"].invoke(user_input)
    formatted_context = combine_docs(retrieved_docs)
    llm_response = ollama_llm_open(user_input, formatted_context, state["history"])
    state["history"].append({"user_input": user_input, "llm_response": llm_response})
    return llm_response


def rag_chain_generate_scenario(state: SessionState):
    retrieved_docs = state["retriever"].invoke("generate scenario")
    formatted_context = combine_docs(retrieved_docs)
    scenario = ollama_llm_generate_scenario(formatted_context)
    return scenario


def rag_chain_eval_scenario(scenario: str, user_response: str, state: SessionState):
    evaluation = ollama_llm_scenario_eval(scenario, user_response)
    state["history"].append(
        {"scenario": scenario, "user_response": user_response, "llm_eval": evaluation}
    )
    return evaluation


# -----------------------------------------------------------------------------
# Gradio UI callbacks
# -----------------------------------------------------------------------------

def handle_file_upload(file, chat_history: List[Tuple[str, str]], state: SessionState):
    """Initialise retriever & context when a lesson‑plan file is uploaded."""
    if file is None:
        return chat_history, state

    retriever, context_text = build_retriever(file.name)
    state.update(
        {
            "retriever": retriever,
            "context": context_text,
            "lesson_plan_name": os.path.basename(file.name),
            "history": [],  # reset chat history on new upload
            "current_scenario": None,
        }
    )

    chat_history.append(
        ("System", f"Lesson plan '{state['lesson_plan_name']}' loaded and indexed!")
    )
    return chat_history, state


def send_message(
    user_input: str, chat_history: List[Tuple[str, str]], state: SessionState
):
    """Handle normal chat flow, scenario generation, and scenario evaluations."""
    if not state.get("retriever"):
        chat_history.append(
            (user_input, "Please upload a lesson plan file before chatting.")
        )
        return chat_history, state

    # Check if the user is asking for a new scenario
    if is_scenario_request(user_input):
        scenario = rag_chain_generate_scenario(state)
        state["current_scenario"] = scenario
        chat_history.append((user_input, "Sure—here is a scenario for you:"))
        chat_history.append(("Scenario", scenario))
        chat_history.append(("System", "Please type your response to the scenario."))
        return chat_history, state

    # If a scenario is awaiting the user's answer, treat this input as the answer
    if state.get("current_scenario"):
        scenario = state["current_scenario"]
        evaluation = rag_chain_eval_scenario(scenario, user_input, state)
        assistant_response = f"Evaluation:\n\n{evaluation}"
        state["current_scenario"] = None
    else:
        assistant_response = rag_chain_open(user_input, state)

    chat_history.append((user_input, assistant_response))
    return chat_history, state


def generate_scenario(chat_history: List[Tuple[str, str]], state: SessionState):
    """Explicit button: create a new scenario and prompt the user to respond."""
    if not state.get("retriever"):
        chat_history.append(
            ("System", "Please upload a lesson plan file before generating a scenario.")
        )
        return chat_history, state

    scenario = rag_chain_generate_scenario(state)
    state["current_scenario"] = scenario
    chat_history.append(("Scenario", scenario))
    chat_history.append(("System", "Please type your response to the scenario."))
    return chat_history, state


def save_history(state: SessionState):
    """Persist the conversation history to disk and return the file path."""
    os.makedirs("history", exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    lesson = state.get("lesson_plan_name", "lesson")
    file_path = f"history/history_{lesson}_{timestamp}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(state["history"], f, indent=4, ensure_ascii=False)
    return file_path


# -----------------------------------------------------------------------------
# Build the Gradio interface
# -----------------------------------------------------------------------------


def build_ui():
    with gr.Blocks(title="ESL Agent for ITAs") as demo:
        gr.Markdown(
            """# ESL Agent for International Teaching Assistants\nUpload a lesson‑plan text file, then chat freely or click **Scenario** to practise responding to realistic classroom situations. The **Save History** button will export your conversation to a JSON file."""
        )

        with gr.Row():
            file_upload = gr.File(label="📄 Upload Lesson Plan (.txt)")
            save_btn = gr.Button("💾 Save History", variant="secondary")
            download_file = gr.File(label="⬇️ Download Chat History")  # ← NEW

        chatbot = gr.Chatbot(height=400)
        user_input = gr.Textbox(label="Your message: (type here and press Enter)")
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            # scenario_btn = gr.Button("🎯 Scenario", variant="secondary")

        # Per‑session state
        session_state = gr.State(
            {
                "retriever": None,
                "context": "",
                "lesson_plan_name": "",
                "history": [],
                "current_scenario": None,
            }
        )

        # Wire events
        file_upload.change(
            handle_file_upload,
            inputs=[file_upload, chatbot, session_state],
            outputs=[chatbot, session_state],
        )
        
        send_btn.click(
            send_message,
            inputs=[user_input, chatbot, session_state],
            outputs=[chatbot, session_state],
        )
        user_input.submit(
            send_message,
            inputs=[user_input, chatbot, session_state],
            outputs=[chatbot, session_state],
        )
        # scenario_btn.click(
        #     generate_scenario,
        #     inputs=[chatbot, session_state],
        #     outputs=[chatbot, session_state],
        # )
        save_btn.click(
            lambda s: save_history(s),
            inputs=session_state,
            outputs=download_file,
        )
        

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_port=7862, share=True, debug=True, server_name="0.0.0.0")
