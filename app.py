import streamlit as st
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentExecutor
from langgraph.graph import StateGraph, START, END
import pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from langchain.tools import Tool
from langchain.typing import TypedDict


st.set_page_config(page_title="Ajay's AI Personal Assistant", page_icon="🤖", layout="centered")

# Title and description
st.title("Ajay's AI Personal Assistant")
st.markdown("Ask me anything — personal, general, weather, or stock-related!")

# Import API keys 
gemini_api_key = st.secrets["gemini"]["api_key"]
pinecone_api_key = st.secrets["pinecone"]["api_key"]
weather_api_key = st.secrets["weather"]["api_key"]
alpha_vantage_api_key = st.secrets["alpha_vantage"]["api_key"]

# Initialize Pinecone
pc = pinecone(api_key=pinecone_api_key)
index_name = "chatbot-index-gemini"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        # dimension of the vector embeddings produced by gemini-embedding-001
        dimension=768, #dimension=768-small, 1536-medium, 3072-large
        metric="cosine",
        # parameters for the free tier index
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Initialize index client
index = pc.Index(name=index_name)

# Initialize vector store retriever
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)

vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# personal info tool
def personal_info_tool(query : str) -> str :
  """ Retrieve Ajay's information and generate natural answer"""
  results = retriever.invoke(query)
  #print(query)

  if not results:
    return "I am not able to find personal information about that"

  context = "\n".join([doc.page_content for doc in results])
  prompt = f"You are Ajay's assistant. Based on the following context, answer the question:\n\n{context}\n\nQuestion: {query}"
  #print(prompt)
  response = llm.invoke(prompt)

  return response.content

# weather_tool
def get_weather_report(location: str) -> str:
    """Fetch current weather for a given location using OpenWeatherMap API."""
    api_key = weather_api_key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"

    response = requests.get(url)
    if response.status_code != 200:
        return f"Couldn't fetch weather for {location}. Please check the location name."

    data = response.json()
    temp = data["main"]["temp"]
    condition = data["weather"][0]["description"]
    return f"The current temperature in {location} is {temp}°C with {condition}."


# stock_tool
def get_stock_price(symbol: str) -> str:
    """Fetch the latest stock price for a given symbol using Alpha Vantage."""
    api_key = alpha_vantage_api_key
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"Error fetching data for {symbol}. Status code: {response.status_code}"

    data = response.json()
    try:
        quote = data["Global Quote"]
        price = quote["05. price"]
        change = quote["10. change percent"]
        return f"{symbol} is currently trading at ${price} ({change} change)."
    except KeyError:
        return f"No data found for symbol: {symbol}"


PersonalInfoRetriever_tools = Tool(name = "PersonalInfoRetriever", func = personal_info_tool, description= "Use this tool to answer any question about Ajay Bellamkonda, including his education, work experience, personal background, or achievements.")

weather_tool = Tool(name="WeatherFetcher", func=get_weather_report, description="Use this tool to get the current weather for any city.")

stock_tool = Tool(name="StockPriceFetcher", func=get_stock_price, description="Use this tool to get the latest stock price for a company. Input should be a stock symbol like AAPL or TSLA.")

tools = [PersonalInfoRetriever_tools,weather_tool,stock_tool]

# Initialize llm 
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)

# Initialize agent
agent_executor = initialize_agent(tools=tools, llm=llm)

# Add nodes and edges
class AgentState(TypedDict):
  input : str
  output : str

def call_agent(state:AgentState):
  result = agent_executor.invoke({"input" : state["input"]})
  return {"output" : result["output"]}

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_agent)
workflow.set_entry_point("agent")
workflow.add_edge("agent",END)


app1 = workflow.compile()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

import streamlit as st
import time

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Chat input box
user_input = st.chat_input("Type your message...")

# Handle user input
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Show loading spinner while generating response
    with st.spinner("Thinking..."):
        time.sleep(1.5)  # Simulate delay — replace with actual backend call

        def get_bot_response(query):
            result = app1.invoke({"input": query})
            return result["output"]

        response = get_bot_response(user_input)

    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

