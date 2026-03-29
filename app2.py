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
ACTIVE_DEMO_POLY_ID = None
current_chat_history_for_rag = None

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

@tool
def get_farm_data():
      global ACTIVE_DEMO_POLY_ID
      weather_data = get_current_weather(ACTIVE_DEMO_POLY_ID)
      soil_data = get_soil_data(ACTIVE_DEMO_POLY_ID)
      weather = json.dumps(weather_data, indent=4)
      soil = json.dumps(soil_data, indent=4)
      return weather+"\n"+soil

@tool
def query_crop_manuals_with_history(query: str) -> str:
    """
    Queries farming manuals to give practical advice for 'how-to' questions.
    This tool uses the current chat history to better understand the context and provide relevant advice.
    """
    global current_chat_history_for_rag
    context = expand_and_retrieve_context(query, current_chat_history_for_rag)
    return context      

langchain_tools = [
    get_farm_data,
    query_crop_manuals_with_history, 
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