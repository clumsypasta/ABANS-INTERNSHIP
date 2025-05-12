# Trade Position Calculator

A system for calculating FIFO and LIFO weighted average prices for open positions based on trade data. The system uses Google's Agent Development Kit (ADK) and Agent-to-Agent (A2A) protocol to create a distributed, modular architecture.

## System Architecture

The system consists of five agents:

1. **Frontend Agent**: Provides a user interface for uploading files and viewing results
2. **Data Agent**: Parses CSV/Excel files into a uniform JSON trade list
3. **Compute Agent**: Nets trades by contract and computes FIFO/LIFO weighted average prices
4. **Pricing Agent**: Fetches real-time prices for marking-to-market (optional)
5. **Visualization Agent**: Generates charts and tables for position data

Each agent is implemented as a separate ADK agent with its own server, and they communicate using the A2A protocol.

## Features

- Parse arbitrary-length trade data from CSV or Excel files
- Net buy/sell trades per contract
- Compute FIFO and LIFO weighted average prices for open positions
- Fetch real-time prices for marking-to-market (optional)
- Generate visualizations of position data

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Starting the System

To start all agents:

```bash
python main.py
```

To start all agents and the Streamlit UI:

```bash
python main.py --streamlit
```

### Accessing the UI

Once the system is running, you can access the Streamlit UI at:

```
http://localhost:8080
```

### Using the A2A API

You can also interact with the agents directly using the A2A API:

- Frontend Agent: `http://localhost:8000`
- Data Agent: `http://localhost:8001`
- Compute Agent: `http://localhost:8002`
- Pricing Agent: `http://localhost:8003`
- Visualization Agent: `http://localhost:8004`

Each agent exposes its capabilities through the A2A protocol, and you can discover them by accessing the agent card at:

```
http://<host>:<port>/.well-known/agent.json
```

## Data Format

The system expects trade data in the following format:

- **Date**: Date of the trade
- **Commodity/Contract**: Name of the contract (e.g., "GOLD", "SILVER")
- **Buy**: Quantity bought (if a buy trade)
- **Buy Average**: Price of the buy trade
- **Sell**: Quantity sold (if a sell trade)
- **Sell Average**: Price of the sell trade
- **Exchange**: Exchange code (e.g., "MCX")
- **Expiry**: Expiry date of the contract
- **Client Code**: Client identifier
- **Strategy**: Strategy name
- **Code**: Strategy code
- **Tagging**: Optional tagging information

## Architecture Details

### A2A Communication Flow

1. **File Upload**: User uploads file → Frontend Agent → sends JSON-RPC `parse_file` request to Data Agent.
2. **Parsing Response**: Data Agent returns `trades: [{contract, date, side, qty, price}, …]`.
3. **Computation**: Frontend Agent issues `compute_positions` to Compute Agent with parsed trades.
4. **Pricing** (if needed): Compute Agent may call Pricing Agent via `get_price` for any contract tickers.
5. **Visualization**: Compute Agent sends position data to Visualization Agent via `plot_positions`.
6. **Render**: Visualization Agent returns chart payloads; Frontend Agent injects into Streamlit.

### FIFO/LIFO Algorithm

For each contract:

1. **Group** all trades by contract.
2. **Sort** by date ascending.
3. **Maintain** two deques:
   - **fifo_deque**: append buys to right, pop from left on sells
   - **lifo_deque**: append buys to right, pop from right on sells
4. **Apply** each trade sequentially:
   - **Buy**: push `{qty, price}` to both deques;
   - **Sell**: decrement deques until `qty_sold` is covered, tracking used quantities and maintaining leftover lots.

After processing all trades, the remaining entries in each deque represent open positions under FIFO and LIFO.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
