import os
import pickle
import faiss
import numpy as np
import requests
import json
import datetime
import time
import mimetypes

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Union, Tuple, Dict, Any

import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from langchain.agents import AgentExecutor, create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
AGROMONITORING_API_KEY = os.environ.get("AGROMONITORING_API_KEY")
RAILWAY_PREDICT_API_URL = "https://tanahpulih-api-production.up.railway.app/predict" # Your CV API URL
DATA_PATH = "data/" # Path to your FAISS index and chunks

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set. Please set it in your environment variables.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set. Please set it in your environment variables.")
if not AGROMONITORING_API_KEY:
    raise ValueError("AGROMONITORING_API_KEY not set. Please set it in your environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

# --- Global Variables and Model Loading ---
llm_rag = genai.GenerativeModel('gemini-flash-latest')
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
index = None
paddy_knowledge = []
polygon_cache: Dict[str, Dict[str, Any]] = {} # Cache for polygon data

# --- Load FAISS Index and Knowledge ---
try:
    with open(os.path.join(DATA_PATH, 'paddy_chunks.pkl'), 'rb') as f:
        paddy_knowledge = pickle.load(f)
    index = faiss.read_index(os.path.join(DATA_PATH, 'paddy_index.faiss'))
    print(f"Successfully loaded {len(paddy_knowledge)} chunks and FAISS index from {DATA_PATH}.")
except FileNotFoundError:
    print(f"ERROR: FAISS index or chunks not found in {DATA_PATH}. Please ensure '{DATA_PATH}paddy_chunks.pkl' and '{DATA_PATH}paddy_index.faiss' exist.")
except Exception as e:
    print(f"ERROR loading FAISS index: {e}")

# --- Helper Functions (from your Colab) ---

# Query Expansion & Retrieval for RAG
def expand_query_with_history(user_query: str, history: List[Dict[str, str]]) -> str:
    history_str = ""
    for msg in history[-6:]:
        speaker = "User" if msg["role"] == "user" else "AI"
        history_str += f"{speaker}: {msg['content']}\n"

    expansion_prompt = f"""
    Given the following conversation history and a new user question that is related Paddy (rice) farming, rewrite the question to be a comprehensive search query for a technical manual.

    Rules:
    - If the user uses pronouns like "it", "they", or "this", replace them with the actual subject (e.g., 'Paddy pests', 'irrigation').
    - Focus on technical keywords (e.g., 'bacterial leaf blight', 'NPK ratio', 'transplanting').
    - Keep the output as a one or three descriptive sentence.

    History:
    {history_str}

    New Question: {user_query}

    Optimized Search Query:"""

    try:
        response = llm_rag.generate_content(expansion_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error during query expansion: {e}")
        return user_query # Fallback to original query

def retrieve_paddy_context(expanded_query: str, k: int = 4) -> List[str]:
    if not index or not paddy_knowledge:
        return ["Error: Knowledge base not loaded."]
    try:
        query_vec = embed_model.encode([expanded_query])
        faiss.normalize_L2(query_vec)
        distances, indices = index.search(np.array(query_vec), k)
        relevant_chunks = [paddy_knowledge[i] for i in indices[0]]
        return relevant_chunks
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return [f"Error retrieving context: {e}"]

def expand_and_retrieve_context(user_input: str, chat_history: List[Dict[str, str]]) -> str:
    expanded_query = expand_query_with_history(user_input, chat_history)
    context_chunks = retrieve_paddy_context(expanded_query)
    return "\n\n".join(context_chunks)

# Agromonitoring API Wrappers
def create_polygon_api(url: str, payload: Dict[str, Any]) -> str | None:
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        poly_data = response.json()
        return poly_data.get('id')
    except requests.exceptions.RequestException as e:
        print(f"Error creating polygon: {e}")
        return None

def get_current_weather_api(poly_id: str) -> Dict[str, Any] | None:
    url = f"https://api.agromonitoring.com/agro/1.0/weather?polyid={poly_id}&appid={AGROMONITORING_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting weather: {e}")
        return None

def get_soil_data_api(poly_id: str) -> Dict[str, Any] | None:
    url = f"http://api.agromonitoring.com/agro/1.0/soil?polyid={poly_id}&appid={AGROMONITORING_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting soil data: {e}")
        return None

def get_satellite_imagery_api(poly_id: str) -> Dict[str, Any] | None:
    end_time = int(time.time())
    start_time = end_time - (30 * 24 * 60 * 60)

    url = f"http://api.agromonitoring.com/agro/1.0/image/search?start={start_time}&end={end_time}&polyid={poly_id}&appid={AGROMONITORING_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        images = response.json()

        best_image = None
        if images:
            for img in images:
                if img.get('cl', 100) < 10: # Use get with default in case 'cl' is missing
                    best_image = img
                    break
            if not best_image:
                best_image = min(images, key=lambda x: x.get('cl', 100))

            if best_image:
                return {
                    "date": datetime.datetime.fromtimestamp(best_image['dt']).strftime('%Y-%m-%d'),
                    "cloud_coverage": best_image.get('cl'),
                    "ndvi_score": best_image.get('stats', {}).get('ndvi'),
                    "ndvi_image_url": best_image.get('image', {}).get('ndvi'),
                    "ndwi_image_url": best_image.get('image', {}).get('ndwi')
                }
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting satellite imagery: {e}")
        return None


# --- LangChain Tools ---

@tool
def get_farm_data(poly_id: str) -> Dict[str, Any]:
    """
    Retrieves current weather, soil temperature, soil moisture, and NDVI health data for a given farm polygon ID.
    It's crucial to provide the correct poly_id for accurate data.
    """
    if poly_id in polygon_cache and (time.time() - polygon_cache[poly_id]['timestamp']) < 300: # Cache for 5 minutes
        print(f"Cache hit for poly_id: {poly_id}")
        return polygon_cache[poly_id]['data']

    weather_data = get_current_weather_api(poly_id)
    soil_data = get_soil_data_api(poly_id)
    satellite_data = get_satellite_imagery_api(poly_id)

    if not weather_data and not soil_data and not satellite_data:
        raise HTTPException(status_code=404, detail=f"No data found for polygon ID: {poly_id}")

    combined_data = {
        "weather": {
            "temperature (Kelvin)": weather_data.get("main", {}).get("temp") if weather_data else None,
            "humidity": weather_data.get("main", {}).get("humidity") if weather_data else None,
            "description": weather_data.get("weather", [{}])[0].get("description") if weather_data else None
        },
        "soil": {
            "t10_temp": soil_data.get("t10") if soil_data else None,
            "moisture": soil_data.get("moisture") if soil_data else None,
            "surface_temp (Kelvin)": soil_data.get("t0") if soil_data else None
        },
        "satellite": satellite_data
    }
    
    polygon_cache[poly_id] = {'data': combined_data, 'timestamp': time.time()}
    return combined_data

# Need to pass chat history to this tool
# Global variable to hold chat history for the RAG tool. In a real app, this would be managed per user session.
# For this example, we'll rely on the chat API to pass it.
current_chat_history_for_rag: List[Dict[str, str]] = []

@tool
def query_crop_manuals_with_history(query: str) -> str:
    """
    Queries farming manuals to give practical advice for 'how-to' questions.
    This tool uses the current chat history to better understand the context and provide relevant advice.
    """
    global current_chat_history_for_rag
    context = expand_and_retrieve_context(query, current_chat_history_for_rag)
    return context

# --- LangChain Agent Setup ---
# Define tools for the agent
langchain_tools = [
    tool(
        func=lambda poly_id: get_farm_data(poly_id),
        args_schema=tool(lambda poly_id: None).parse_json_schema(
            {"poly_id": {"type": "string"}}
        ),
        name="get_farm_data",
        description="Retrieves current weather, soil temperature, soil moisture, and NDVI health data for a given farm polygon ID. It's crucial to provide the correct poly_id for accurate data."
    ),
    tool(
        func=query_crop_manuals_with_history,
        args_schema=tool(query_crop_manuals_with_history).parse_json_schema(
            {"query": {"type": "string"}}
        ),
        name="query_crop_manuals_with_history",
        description="Queries farming manuals to give practical advice for 'how-to' questions. This tool uses the current chat history to better understand the context and provide relevant advice."
    )
]

# System prompt for the agent
system_prompt_text = """
You are 'AgriGuard', a helpful AI assistant for farmers.

IMPORTANT GUIDELINES:
- ONLY answer questions related to farming, crops, soil, and weather.
- If a question is unrelated to farming, politely respond: "I focus on farming. Please ask me about your crops."
- When the user asks about their farm, use the 'get_farm_data' tool FIRST to get the latest readings for their specified farm polygon ID.
- If the satellite data shows an anomaly (e.g., low NDVI), mention it clearly in your response.
- If a user asks a 'how-to' question or asks for advice, use the 'query_crop_manuals_with_history' tool to find the answer in the farming manuals.
- When giving advice, always combine information from 'get_farm_data' and 'query_crop_manuals_with_history' for a comprehensive answer.
- Speak in plain, simple language. Avoid jargon.
- If the user's input suggests a potential plant disease or a need for visual inspection, ask them to upload a photo. DO NOT attempt to analyze the photo yourself here; you will ask the user to trigger a separate process for that.
"""

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini", # Use a cost-effective model for development
    temperature=0
)

# Agent Executor
agent_executor = create_agent(
    llm,
    langchain_tools,
    prompt_template
)

# --- FastAPI App ---
app = FastAPI()

# CORS Middleware to allow frontend to connect (adjust origins as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.post("/api/v1/farm/polygon", response_model=Dict[str, str])
async def create_farm_polygon(request: Dict[str, Any]):
    """
    API to send coordinates and create a farm polygon.
    Expects: {"name": "Farm Name", "coordinates": [...]}
    """
    payload = {
        "name": request.get("name", "Unnamed Farm"),
        "geo_json": {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [request.get("coordinates")] # Ensure coordinates are nested correctly
            }
        }
    }
    poly_id = create_polygon_api(f"https://api.agromonitoring.com/agro/1.0/polygons?appid={AGROMONITORING_API_KEY}", payload)
    if poly_id:
        return {"status": "success", "poly_id": poly_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to create farm polygon.")

@app.get("/api/v1/farm/data", response_model=Dict[str, Any])
async def get_farm_data_endpoint(poly_id: str):
    """
    API to get satellite, weather, and soil data for a given polygon ID.
    """
    try:
        data = get_farm_data(poly_id)
        if data is None or (not any(data.values())): # Check if all values are None/empty
             raise HTTPException(status_code=404, detail=f"No data found for polygon ID: {poly_id}")
        
        # Structure the response to match your example (slightly simplified for dynamic data)
        response_data = {
            "weather": data.get("weather", {}),
            "soil": data.get("soil", {}),
            "satellite": data.get("satellite", {})
        }
        
        # Add NDVI score if available
        if response_data["satellite"] and response_data["satellite"].get("ndvi_score") is not None:
             response_data["satellite"]["ndvi"] = response_data["satellite"]["ndvi_score"] # Map to your desired key name
        
        return response_data

    except HTTPException as e:
        raise e # Re-raise HTTPException to return its status code and detail
    except Exception as e:
        print(f"Error in /api/v1/farm/data: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred while fetching farm data: {e}")

@app.post("/api/v1/chat", response_model=Dict[str, Any])
async def chat_endpoint(request: Dict[str, Any]):
    """
    API for conversation, using LangChain agent with RAG and satellite data tools.
    Expects: {"poly_id": "...", "message": "...", "chat_history": [...]}
    """
    user_message = request.get("message")
    poly_id = request.get("poly_id")
    chat_history = request.get("chat_history", [])

    if not user_message or not poly_id:
        raise HTTPException(status_code=400, detail="Missing 'message' or 'poly_id' in request.")

    # Set the global chat history for the RAG tool
    global current_chat_history_for_rag
    current_chat_history_for_rag = chat_history

    # Prepare messages for the agent executor
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            messages.append({"role": "assistant", "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        # Invoke the agent
        # The agent needs to know about the poly_id to use the get_farm_data tool.
        # We can't directly pass poly_id to the agent's 'input' if it's expecting a string message.
        # A common pattern is to prepend instructions to the user message.
        
        # Check if the message already contains the poly_id implicitly or explicitly
        # If not, instruct the agent to use the provided poly_id.
        # NOTE: This is a simplification. A more robust solution might involve an agent that
        # explicitly takes poly_id as an argument or a preliminary step to fetch data.
        
        # Let's assume the user message is clear enough for the agent to *eventually* call get_farm_data
        # IF it decides to. The current prompt structure needs the agent to decide.
        # For this example, we'll modify the input slightly for better tool invocation.
        
        # A better way is to have the agent explicitly take poly_id, or use a custom tool wrapper.
        # For simplicity here, let's make the prompt slightly more explicit about needing poly_id.
        
        # To ensure get_farm_data is called, we can try to instruct the agent.
        # The current system prompt guides it to check for farm data first.
        
        # We need to ensure the agent knows the poly_id for get_farm_data.
        # Let's modify the user's input to include the poly_id context for the agent.
        contextual_user_message = f"Farm ID: {poly_id}\nUser Message: {user_message}"

        response = agent_executor.invoke({
            "messages": messages,
            "input": contextual_user_message # Use the contextual message for the agent
        })

        assistant_reply = response["messages"][-1].content

        # Clean up global chat history for the RAG tool.
        # In a real app, chat history would be managed per user session.
        current_chat_history_for_rag = [] 

        return {"reply": assistant_reply}

    except Exception as e:
        print(f"Error during chat interaction: {e}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred during chat: {e}")

@app.post("/api/v1/vision/predict", response_model=Dict[str, Any])
async def predict_disease_from_image(file: UploadFile = File(...)):
    """
    API to upload a plant image and predict diseases using an external service.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    
    # Check file type (basic check)
    content_type, _ = mimetypes.guess_type(file.filename)
    if not content_type or not content_type.startswith('image'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        files_payload = {
            "file": (file.filename, await file.read(), file.content_type)
        }
        response = requests.post(RAILWAY_PREDICT_API_URL, files=files_payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Railway predict API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get prediction from external service: {e}")
    except Exception as e:
        print(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred while processing the image: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    print(f"Agromonitoring API Key set: {AGROMONITORING_API_KEY is not None}")
    print(f"Google API Key set: {GOOGLE_API_KEY is not None}")
    print(f"OpenAI API Key set: {OPENAI_API_KEY is not None}")
    print(f"Data path for FAISS: {os.path.abspath(DATA_PATH)}")
    uvicorn.run(app, host="0.0.0.0", port=8000)