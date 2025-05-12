# AI Assistant Agent

This agent uses Google's Gemini AI to analyze trade data and answer user questions about their positions and trades.

## API Key Setup

To use the AI Assistant Agent, you need a Google API key for Gemini. Here's how to get one:

1. Go to the [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click on "Create API Key"
4. Copy the generated API key

## Using the API Key

You can provide the API key in one of these ways (listed in order of preference):

### 1. Using a .env File (Recommended)

Create a `.env` file in the root directory of the project with your API key:

1. Copy the `.env.sample` file to a new file named `.env`
2. Edit the `.env` file and replace `your_api_key_here` with your actual API key:

```
GOOGLE_API_KEY=your_actual_api_key_here
```

This is the most convenient method as you only need to set it once.

### 2. Environment Variable

Set the `GOOGLE_API_KEY` environment variable before running the application:

```bash
# On Windows
set GOOGLE_API_KEY=your_api_key_here

# On macOS/Linux
export GOOGLE_API_KEY=your_api_key_here
```

### 3. Command Line Argument

Pass the API key as a command line argument when starting the application:

```bash
# Running the main application
python main.py --streamlit --api-key your_api_key_here
```

### 4. Direct to the Assistant Agent

If you're running the Assistant Agent separately:

```bash
python -m agents.assistant.server --api-key your_api_key_here
```

## Features

The AI Assistant can:

- Answer questions about your trade data
- Explain FIFO and LIFO calculations
- Provide insights about your positions
- Analyze trading patterns
- Help you understand your P&L

## Example Questions

- "What are my largest positions?"
- "Explain the difference between FIFO and LIFO"
- "How many trades do I have for GOLD?"
- "What's my average purchase price for SILVER?"
- "Which position has the biggest unrealized P&L?"
