# 🌾 TanahPulih — Backend API

> AI-powered paddy farm monitoring backend for Malaysian smallholder farmers.  
> Built for **PutraHack 2026 — Food Security Hackathon**.

---

## 📖 Overview

TanahPulih's backend is a **FastAPI** server that acts as the central orchestration layer for the entire platform. It connects satellite data, a RAG-powered chatbot agent, and a computer vision disease detection model into a single unified API.

---

## 🏗️ Architecture

```
Farmer (Frontend PWA)
        │
        ▼
  FastAPI Backend  ──────────────────────────────────────────┐
        │                                                      │
        ├── /farm/polygon  ──► Agromonitoring API (register farm polygon)
        │                                                      │
        ├── /farm/data  ──────► Agromonitoring API (weather + soil + NDVI)
        │                                                      │
        ├── /chat  ───────────► LangChain Agent (GPT-4o-mini)  │
        │                            ├── Tool: get_farm_data   │
        │                            └── Tool: query_crop_manuals (RAG)
        │                                      └── FAISS + Gemini Flash
        │                                                      │
        └── /vision/predict  ─► Railway CV API (EfficientNet-B0)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- API keys for: Google (Gemini), OpenAI, and Agromonitoring

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourteam/tanahpulih
cd tanahpulih/backend

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Then fill in your keys (see below)

# 5. Add required data files
# Place paddy_chunks.pkl and paddy_index.faiss inside the /data folder

# 6. Run the server
uvicorn app:app --reload
```

The API will be live at `http://localhost:8000`.

---

## 🔑 Environment Variables

Create a `.env` file in the root of the backend directory with the following:

```env
GOOGLE_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
AGROMONITORING_API_KEY=your_agromonitoring_api_key
```

| Variable                 | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| `GOOGLE_API_KEY`         | Used for Gemini Flash (RAG query expansion)              |
| `OPENAI_API_KEY`         | Used for GPT-4o-mini (chatbot agent) and text embeddings |
| `AGROMONITORING_API_KEY` | Used for satellite imagery, weather, and soil data       |

---

## 📡 API Endpoints

### `POST /api/v1/farm/polygon`
Registers a farm's geographic boundary on Agromonitoring and returns a `poly_id` used for all subsequent data queries.

**Request Body:**
```json
{
  "name": "Sawah Pak Ali",
  "coordinates": [[101.5, 3.1], [101.6, 3.1], [101.6, 3.2], [101.5, 3.2], [101.5, 3.1]]
}
```

**Response:**
```json
{
  "status": "success",
  "poly_id": "abc123xyz"
}
```

---

### `GET /api/v1/farm/data?poly_id={poly_id}`
Fetches current satellite, weather, and soil data for a registered farm polygon.

**Response:**
```json
{
  "weather": { "temp": 30.5, "humidity": 85 },
  "soil": { "t0": 28.3, "moisture": 0.42 },
  "satellite": {
    "date": "2026-03-28",
    "cloud_coverage": 12,
    "ndvi_score": 0.61,
    "ndvi_image_url": "https://...",
    "ndwi_image_url": "https://..."
  }
}
```

---

### `POST /api/v1/chat`
Main conversational endpoint. Powered by a **LangChain agent (AgriGuard)** that autonomously decides when to fetch farm data or query the paddy knowledge base.

**Request Body:**
```json
{
  "poly_id": "abc123xyz",
  "message": "Why are the leaves turning yellow?",
  "chat_history": [
    { "role": "user", "content": "Hello" },
    { "role": "assistant", "content": "Hello! How can I help?" }
  ]
}
```

**Response:**
```json
{
  "reply": "Based on your current farm data, the NDVI score is 0.61 which is within normal range. However, yellowing leaves could indicate nitrogen deficiency or early-stage Tungro virus..."
}
```

---

### `POST /api/v1/vision/predict`
Accepts a plant image upload and forwards it to the CV model hosted on Railway for disease classification.

**Request:** `multipart/form-data` with a `file` field (image).

**Response:**
```json
{
  "disease": "Rice Blast",
  "confidence": 0.91,
  "severity": "moderate"
}
```

> ⚠️ If confidence is below **0.72**, the app will prompt the farmer to retake the photo.

---

## 🤖 AgriGuard — The AI Agent

The chatbot is built using a **LangChain Tool-Calling Agent** backed by `gpt-4o-mini`. It has access to two tools:

| Tool                              | Description                                                           |
| --------------------------------- | --------------------------------------------------------------------- |
| `get_farm_data`                   | Fetches live weather, soil, and NDVI data for the active farm polygon |
| `query_crop_manuals_with_history` | Performs RAG search over paddy farming manuals using FAISS            |

### RAG Pipeline

1. **Query Expansion** — Gemini Flash rewrites the user's question using the last 6 messages of chat history to produce a better technical search query
2. **Vector Search** — The expanded query is embedded (`text-embedding-3-small`) and searched against a local FAISS index
3. **Context Injection** — Top 4 relevant chunks from the paddy knowledge base are passed to the agent as context

---

## 🧠 CV Model

The disease detection model is **EfficientNet-B0** fine-tuned on the Paddy Doctor dataset:

| Detail               | Info                                                           |
| -------------------- | -------------------------------------------------------------- |
| Architecture         | EfficientNet-B0 (frozen base + fine-tuned head)                |
| Dataset              | Paddy Doctor — 10,407 training images                          |
| Classes              | Healthy, Rice Blast, Brown Spot, Bacterial Leaf Blight, Tungro |
| Validation Accuracy  | ~95%                                                           |
| Confidence Threshold | 0.72 (below this, farmer is asked to retake photo)             |
| Deployment           | Hosted on Railway (`/predict` endpoint)                        |

---

## 📦 Dependencies

```
fastapi==0.111.0
uvicorn==0.30.1
python-multipart==0.0.9
numpy==1.26.4
faiss-cpu==1.8.0
langchain==0.2.5
langchain-community==0.2.5
langchain-core==0.2.9
langchain-openai==0.1.8
google-generativeai==0.7.2
requests==2.32.3
pydantic==2.7.4
pydantic-settings==2.3.3
```

---

## 📁 Required Data Files

Place these in the `/data` directory before running the server:

```
data/
├── paddy_chunks.pkl     # Serialized text chunks from paddy farming manuals
└── paddy_index.faiss    # Pre-built FAISS vector index
```

These files are generated by the RAG indexing pipeline (see `/ml` directory).

---

## ⚠️ Known Limitations

- `ACTIVE_DEMO_POLY_ID` is stored as a **global variable** — safe for single-user demos but will cause race conditions with multiple concurrent users. Future fix: move to session/request-scoped state.
- CORS is currently set to `allow_origins=["*"]` — restrict this to your frontend URL before production deployment.

---

## 🗺️ Roadmap

- [ ] Per-session polygon ID management
- [ ] Supabase integration for persistent farm scan history
- [ ] Telegram Bot alert dispatcher
- [ ] Offline-capable TFLite model for edge deployment
- [ ] Extend CV model to support chilli and other Malaysian crops

---

*Built with ❤️ for Malaysian farmers — PutraHack 2026*