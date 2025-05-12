# Trade Position Calculator

A system for calculating FIFO and LIFO weighted average prices for open positions based on trade data. The system uses a modular architecture with multiple agents for different functionalities.

## Live Demo

You can access the live demo at: [Streamlit Cloud App](https://your-app-url-here)

## System Architecture

The system consists of six agents:

1. **Frontend Agent**: Provides a user interface for uploading files and viewing results
2. **Data Agent**: Parses CSV/Excel files into a uniform JSON trade list
3. **Compute Agent**: Nets trades by contract and computes FIFO/LIFO weighted average prices
4. **Pricing Agent**: Fetches real-time prices for marking-to-market (optional)
5. **Visualization Agent**: Generates charts and tables for position data
6. **AI Assistant Agent**: Provides natural language insights about trade data

## Features

- Parse arbitrary-length trade data from CSV or Excel files
- Net buy/sell trades per contract
- Compute FIFO and LIFO weighted average prices for open positions
- Fetch real-time prices for marking-to-market (optional)
- Generate visualizations of position data
- AI-powered insights and answers about your trade data

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your Google API key for the AI Assistant:
   - Create a `.env` file in the root directory with your API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
   - Or set it as an environment variable

## Usage

### Running Locally

To start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

### Accessing the UI

Once the app is running, you can access it at:

```
http://localhost:8501
```

## Data Format

The system expects trade data in the following format:

| Date | Commodity | Buy | Buy Average | Sell | Sell Average | Exchange | Expiry | Client Code | Strategy | Code | Remarks | Tagging |
|------|-----------|-----|------------|------|--------------|----------|--------|-------------|----------|------|---------|---------|
| 2023-01-01 | GOLD | 10 | 1900.50 |  |  | COMEX | 2023-12-31 | 12345 | HEDGE | ABC123 | Initial purchase | Portfolio A |
| 2023-01-15 |  |  |  | 5 | 1950.25 | COMEX | 2023-12-31 | 12345 | HEDGE | ABC123 | Partial sale | Portfolio A |

## AI Assistant

The AI Assistant can answer questions about your trade data, such as:

- "What are my largest positions?"
- "Explain the difference between FIFO and LIFO"
- "How many trades do I have for GOLD?"
- "What's my average purchase price for SILVER?"
- "Which position has the biggest unrealized P&L?"

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- AI capabilities powered by [Google Gemini](https://ai.google.dev/)
