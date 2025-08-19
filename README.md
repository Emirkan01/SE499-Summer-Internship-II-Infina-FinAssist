# FinAssist 💹

**Bilingual (TR/EN) finance chatbot** built with **Gradio**, **LangGraph**, **Google Gemini 2.5 Flash-Lite**, and **Turkish BERT**.  
Includes a **Forecasts tab** that displays BIST30 forecast PNGs directly from your local folder (no Excel uploads needed).

---

## 🚀 Features
- 🤖 Chatbot with **TR/EN support**
- 📊 **BIST30 forecasts** displayed from local `./forecasts/` folder
- 🔍 **RAG** (retrieval-augmented generation) with FAISS + multilingual embeddings
- 🧮 **Finance calculators** (e.g., credit card interest, budgeting)
- ⚡ Powered by **Gemini 2.5 Flash-Lite** (Google Generative AI)

---

## ⚙️ Installation

Clone this repo (or download ZIP), then install dependencies:

```bash
pip install -U \
  gradio langchain langgraph langchain-core langchain-google-genai \
  sentence-transformers faiss-cpu numpy pydantic \
  google-generativeai python-dotenv pillow
