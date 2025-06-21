# Datateam

This project named "Datateam" is a DataOps AI Agent — a system that automates the full lifecycle of dataset generation and processing. It sits between raw data and model training, doing the heavy lifting: collecting, cleaning, augmenting, validating, and preparing high-quality datasets.

It can perform the following actions:
1. data collection from web
2. Pre-processing (Cleaning, formatting, parsing, normalizing)
3. Synthetic data generation
4. Data validation
5. Taxonomy generation
6. Data augmentation
7. Output generation (e.g., CSV, JSON, etc.)

## Prerequisites

- Python 3.12 or later (with Tkinter support)
- Node.js and npm (for Bright Data MCP)
- Google Gemini API key
- Bright Data account with active Web Unblocker API zone (For Browser capabilities, Scraping Browser zone is required as well)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/byjit/datateam.git
cd datateam
```

### 2. Create and activate a virtual environment

```bash
uv venv
source .venv/bin/activate
```

### 3. Install the required packages
```bash
uv pip install google-adk google-generativeai python-dotenv
```

### 4. Install Bright Data MCP package

```bash
npm install -g @brightdata/mcp
```

### 5. Setup Kaggle 

Download your kaggle credentials json inside `~/.kaggle/kaggle.json` 

### 6. Set up environment variables

Create a `.env` file in the root directory by copying the `.env.template`:

```bash
cp .env.template .env
```

Then, edit the `.env` file and add your Google Gemini API key:

```
GOOGLE_GENAI_USE_VERTEXAI="False"
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
"API_TOKEN": "YOUR_BRIGHT_DATA_API_TOKEN",
"WEB_UNLOCKER_ZONE": "unblocker",
"BROWSER_AUTH": "brd-customer-YOUR_CUSTOMER_ID-zone-scraping_browser:YOUR_PASSWORD"
```

## Running the Agent with ADK Web Interface

### 1. Start the ADK Web Server
```bash
adk web
```

This will start a local web server, typically at `http://localhost:8000`.

### 2. Access the Web Interface

Open your browser and navigate to `http://localhost:8000` to interact with your agent through the ADK web interface.