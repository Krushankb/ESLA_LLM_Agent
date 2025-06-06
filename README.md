# 🚀 ESL LLM Agent

Welcome to **ESL_LLM_Agent** – a locally running, AI-powered ESL chatbot using LLaMA 3 and Langchain. 

Click here to read our research paper and learn about the behind the scenes! 

[Link](https://drive.google.com/file/d/1VKDTINOi6yM9DgUz60NqqP_Vb1Zc5cMU/view?usp=sharing)

To get started on running our repo, follow these six simple steps!

---

## 1. 📦 Install Dependencies

Make sure you have Python installed, then install the required Python packages:

```bash
pip install -r requirements.txt
```
## 2. 🧠 Download Ollama
You’ll need to install Ollama, a local large language model runner.

👉 Download Ollama here: https://ollama.com/download

Follow the installation instructions for your operating system.

## 3. 📥 Pull the LLaMA 3 Model
Once Ollama is installed and running, open your terminal and run:

`ollama pull llama3`
This will download the LLaMA 3 model locally so it can be used by the chatbot.

## 4. 🖥️ Run in Terminal
To use the chatbot directly from your terminal interface, run:

`python3 llm_test.py`
This will start a text-based interaction with the chatbot.

## 5. 🌐 Run on a Local Web Server
If you prefer a browser-based interface, run:

`python3 llm_test_2.py`
Open the displayed localhost URL in your browser.

💡 Tip: For the best local development experience, we recommend using the Five Server VSCode extension. It provides a simple live-reload server on localhost.

## 6. 💬 Enjoy Chatting!
You’re all set!
Start a conversation with your ESL chatbot, either through the terminal or your browser.

This project is perfect for:

Practicing English conversation

Exploring local LLM tools

Learning how to integrate Langchain with Ollama

## 🔧 Tech Stack
Langchain – modular framework for LLM applications

Chroma DB – embedding-based vector store

Ollama – local LLM engine

LLaMA 3 – Meta’s powerful language model
##
Feel free to fork this repo, contribute, or reach out with ideas and improvements.
Happy chatting! 🎉
