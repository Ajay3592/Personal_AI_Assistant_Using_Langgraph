# 🧠 AI Personal Assistant Chatbot
URL : https://ajaypersonalbot.streamlit.app/
An intelligent, interactive personal assistant chatbot built using [LangGraph](https://www.langgraph.dev/), powered by Google Gemini and Pinecone vector database. This assistant can answer both general and personal questions, fetch live weather and stock data.

## 🚀 Features

- **Personalized Responses**  
  Retrieves answers to personal questions using your own data stored in Pinecone, embedded via Google Gemini's `embedding-001` model.

- **General Knowledge**  
  Uses Gemini 2.0 Flash to answer regular queries directly via LLM.

- **Live Weather Tool 🌦️**  
  Fetches current weather reports based on user queries(openweather).

- **Stock Price Tool 📈**  
  Provides real-time stock market data for requested tickers(alphavantage).

- **Interactive UI**  
  Deployed using Streamlit for a responsive and user-friendly experience.

## 🧰 Tech Stack

| Component        | Technology Used                          |
|------------------|-------------------------------------------|
| LLM              | Google Gemini 2.0 Flash                   |
| Embeddings       | Google Gemini `embedding-001`             |
| Framework        | LangGraph                                 |
| Vector Database  | Pinecone                                   |
| UI               | Streamlit                                 |
| Tools            | Custom Weather & Stock Price APIs         |


