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
from langchain_openai import OpenAIEmbeddings

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain.agents.agent import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Union, Tuple

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
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
index = None
paddy_knowledge = []
polygon_cache: Dict[str, Dict[str, Any]] = {} # Cache for polygon data
ACTIVE_DEMO_POLY_ID = None
current_chat_history_for_rag = None

def load_faiss():
    global index, paddy_knowledge
    if index is None:
        with open(os.path.join(DATA_PATH, 'paddy_chunks.pkl'), 'rb') as f:
            paddy_knowledge = pickle.load(f)
        index = faiss.read_index(os.path.join(DATA_PATH, 'paddy_index.faiss'))
        print("FAISS loaded")

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
    load_faiss()
    if not index or not paddy_knowledge:
        return ["Error: Knowledge base not loaded."]
    try:
        query_vec = embed_model.embed_query(expanded_query)
        query_vec = np.array([query_vec]).astype("float32")
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
def create_polygon_api(url: str, payload: Dict[str, Any]) -> Union[str,None]:
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        poly_data = response.json()
        return poly_data.get('id')
    except requests.exceptions.RequestException as e:
        print(f"Error creating polygon: {e}")
        return None

def get_current_weather_api(poly_id: str) -> Union[Dict[str, Any],None]:
    url = f"https://api.agromonitoring.com/agro/1.0/weather?polyid={poly_id}&appid={AGROMONITORING_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting weather: {e}")
        return None

def get_soil_data_api(poly_id: str) -> Union[Dict[str, Any],None]:
    url = f"http://api.agromonitoring.com/agro/1.0/soil?polyid={poly_id}&appid={AGROMONITORING_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting soil data: {e}")
        return None

def get_satellite_imagery_api(poly_id: str) -> Union[Dict[str, Any],None]:
    end_time = int(time.time())
    start_time = end_time - (60 * 24 * 60 * 60)

    url = f"http://api.agromonitoring.com/agro/1.0/image/search?start={start_time}&end={end_time}&polyid={poly_id}&appid={AGROMONITORING_API_KEY}"
    print(url)
    try:
        response = requests.get(url)
        response.raise_for_status()
        images = response.json()
        print(images)
        best_image = None
        if images:
            print("SOOMETHIONG")
            for img in images:
                if img.get('cl', 100) < 40: # Use get with default in case 'cl' is missing
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

def get_farm_data_for_api(poly_id: str):
    """
    Fetches raw data and returns it as a structured dictionary.
    """
    # Call your API functions (assuming these return dicts)
    weather_data = get_current_weather_api(poly_id)
    soil_data = get_soil_data_api(poly_id)
    satellite_imagery_data = get_satellite_imagery_api(poly_id)

    # Return as a dictionary, NOT a string
    return {
        "weather": weather_data,
        "soil": soil_data,
        "satellite": satellite_imagery_data
    }

@tool
def get_farm_data():
      """Returns the current weather, soil temperature, soil moisture, and NDVI health data for the user's farm."""
      global ACTIVE_DEMO_POLY_ID
      weather_data = get_current_weather_api(ACTIVE_DEMO_POLY_ID)
      soil_data = get_soil_data_api(ACTIVE_DEMO_POLY_ID)
      satellite_imagery_data = get_satellite_imagery_api(ACTIVE_DEMO_POLY_ID)
      weather = json.dumps(weather_data, indent=4)
      soil = json.dumps(soil_data, indent=4)
      satellite_imagery = json.dumps(satellite_imagery_data, indent=4)
      return weather+"\n"+soil+"\n"+satellite_imagery

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
You are 'AgriGuard', an empathetic, highly knowledgeable AI assistant specifically designed for paddy (rice) crop farmers.

IMPORTANT GUIDELINES:
- SCOPE: ONLY answer questions related to farming, paddy crops, soil, and weather. If a question is unrelated, politely respond: "I focus on farming and crop management. Please ask me about your fields."
- TONE: Speak in plain, simple language. Avoid highly technical jargon. Be reassuring and supportive, as crop issues can be stressful for farmers.

TOOL USAGE:
- FARM CONTEXT: When the user asks about their farm's current status, use the 'get_farm_data' tool FIRST to fetch the latest readings using their polygon ID.
- ANOMALIES: If the satellite data from 'get_farm_data' shows an anomaly (e.g., very low soil moisture, low NDVI/crop health), mention it clearly but gently, and suggest the next steps.
- RAG / MANUALS: If a user asks a 'how-to' question, asks for disease treatment, or needs general advice, use the 'query_crop_manuals_with_history' tool to find the exact, safe answer. NEVER guess; always rely on the manuals to prevent hallucinations.
- COMBINING DATA: Always try to synthesize the current farm conditions (weather, soil) with the manual's advice for a comprehensive answer.

HANDLING IMAGES & VISION AI:
- Sometimes the user's prompt will contain hidden context about an image they uploaded. Example: "[System Notice: The user uploaded a photo of a plant. The Vision AI detected 'hispa' with 87.5% confidence.] User message: What should I do?"
- IF CONFIDENCE IS HIGH (Above 60%): Trust the Vision AI. Look up the detected disease using your 'query_crop_manuals_with_history' tool and give step-by-step treatment advice.
- IF CONFIDENCE IS LOW (Below 60%): Tell the user the AI suspects the disease but isn't entirely sure. Ask them to upload a clearer, closer picture of the affected leaves/stems.
- MISSING VISUALS: If the user's text suggests a potential plant disease (e.g., "my leaves are turning brown") but they haven't uploaded a photo, ask them to upload a picture so AgriGuard can analyze it. DO NOT attempt to diagnose it based purely on a short text description.
"""

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini", # Use a cost-effective model for development
    temperature=0
)
# Define the prompt correctly for a Tool Calling Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text), # Your system prompt text
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
# 1. Create the Agent (this is the "brain" that makes decisions)
agent = create_tool_calling_agent(llm, langchain_tools, prompt)

# 2. Create the AgentExecutor (this is the "engine" that actually calls the tools)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=langchain_tools, 
    verbose=True, 
    handle_parsing_errors=True
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

@app.get("/api/v1/farm/data", response_model=Dict[str, Any])
async def get_farm_data_endpoint(poly_id: str):
    """
    API to get satellite, weather, and soil data for a given polygon ID.
    """
    try:
        data = get_farm_data_for_api(poly_id)
        if data is None or (not any(data.values())): # Check if all values are None/empty
             raise HTTPException(status_code=404, detail=f"No data found for polygon ID: {poly_id}")
        
        # Structure the response to match your example (slightly simplified for dynamic data)
        response_data = {
            "weather": data.get("weather", {}),
            "soil": data.get("soil", {}),
            "satellite": data.get("satellite", {})
        }
       
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
    global ACTIVE_DEMO_POLY_ID
    ACTIVE_DEMO_POLY_ID=poly_id
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

        response = agent_executor.invoke({
            "input": user_message,
            "chat_history": messages
        })

        return {"reply": response["output"]}

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

