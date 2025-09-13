# ChatCSV
🧠 CSV AI (Groq) — Interact, Analyze & Summarize CSV Files with AI




Welcome to CSV AI (Groq) — a powerful and interactive web application that lets you chat with, analyze, and summarize CSV files using advanced AI models from Groq. Unlock insights from your data without writing a single line of code!

⚡ Features

Chat with CSV
Ask natural language questions about your CSV files. Get precise answers backed by your data.

Summarize CSV
Automatically generate concise summaries of large CSV files. Perfect for quick overviews.

Analyze CSV
Perform data analysis using AI-powered agents. Generate insights and explore your data interactively.

API Key Validation
Built-in support to validate your Groq API key safely and instantly.

Multi-Model Support
Use powerful models like:

gemma2-9b-it

llama-3.3-70b-versatile

mixtral-8x7b-32768

meta-llama/llama-guard-4-12b

Customizable Parameters
Adjust Temperature and Top_P for creative or precise responses.

Efficient Vector Search
Converts CSV content into embeddings for fast and accurate similarity search.

Streaming Responses
AI answers appear live as they are generated — just like chatting with a real assistant!

🛠️ Technologies Used

Streamlit
 — Interactive UI for web apps

LangChain
 — Building AI chains & agents

Groq AI Models
 — High-performance language models

FAISS
 — Vector search for embedding-based retrieval

HuggingFace Embeddings
 — Convert text into vector representations

Pandas
 — CSV data handling and manipulation

🚀 Installation & Setup

Clone this repository

git clone https://github.com/yourusername/csv-ai-groq.git
cd csv-ai-groq


Install dependencies

pip install -r requirements.txt


Run the app

streamlit run app.py


Enter your Groq API key in the sidebar to get started!

📂 Usage

Upload a CSV file via the sidebar or main page.

Select one of the functionalities:

Chat with CSV

Summarize CSV

Analyze CSV

Adjust model parameters (Temperature, Top_P) if needed.

Start interacting with your data!

🎯 Why CSV AI?

No coding required — interact directly with your data.

Supports large CSVs — splits data into chunks for accurate AI understanding.

Safe & reliable — validates API keys and handles errors gracefully.

Interactive & intuitive — chat interface for a smooth user experience.

💡 Tips

Use smaller chunks for faster summaries with large CSVs.

Adjust temperature for more creative responses (higher = more creative, lower = more precise).

Validate your Groq API key before starting to avoid authentication issues.


⚖️ License

This project is MIT licensed — feel free to modify and use it for personal or commercial purposes.

🙌 Contributing

Fork the repo

Create a feature branch (git checkout -b feature-name)

Commit your changes (git commit -m "Add feature")

Push (git push origin feature-name)

Open a Pull Request
