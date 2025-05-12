import os
import sys
import logging
import argparse
import subprocess
import time
import signal
import threading
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Agent configurations
AGENTS = [
    {
        "name": "Data Agent",
        "module": "agents.data.server",
        "host": "localhost",
        "port": 8001
    },
    {
        "name": "Compute Agent",
        "module": "agents.compute.server",
        "host": "localhost",
        "port": 8002
    },
    {
        "name": "Pricing Agent",
        "module": "agents.pricing.server",
        "host": "localhost",
        "port": 8003
    },
    {
        "name": "Visualization Agent",
        "module": "agents.visualization.server",
        "host": "localhost",
        "port": 8004
    },
    {
        "name": "AI Assistant Agent",
        "module": "agents.assistant.server",
        "host": "localhost",
        "port": 8005
    },
    {
        "name": "Frontend Agent",
        "module": "agents.frontend.server",
        "host": "localhost",
        "port": 8000
    }
]

# Global variables
processes = []
stop_event = threading.Event()


def start_agent(agent_config):
    """Start an agent in a separate process."""
    cmd = [
        sys.executable,
        "-m",
        agent_config["module"],
        "--host",
        agent_config["host"],
        "--port",
        str(agent_config["port"])
    ]

    # Add API key for AI Assistant Agent if provided
    if agent_config["name"] == "AI Assistant Agent" and "api_key" in agent_config:
        cmd.extend(["--api-key", agent_config["api_key"]])

    logger.info(f"Starting {agent_config['name']} on {agent_config['host']}:{agent_config['port']}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    return process


def log_output(process, agent_name):
    """Log the output of a process."""
    while not stop_event.is_set():
        output = process.stdout.readline()
        if output:
            logger.info(f"[{agent_name}] {output.strip()}")

        error = process.stderr.readline()
        if error:
            logger.error(f"[{agent_name}] {error.strip()}")

        if process.poll() is not None:
            break


def start_streamlit():
    """Start the Streamlit app."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "agents/frontend/app.py",
        "--server.port",
        "8080"
    ]

    logger.info("Starting Streamlit app on port 8080")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    return process


def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully shut down all processes."""
    logger.info("Shutting down...")
    stop_event.set()

    for process in processes:
        if process.poll() is None:
            process.terminate()

    # Wait for processes to terminate
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    logger.info("All processes terminated")
    sys.exit(0)


def main():
    """Main function to start all agents."""
    parser = argparse.ArgumentParser(description="Start the Calculation Agent system")
    parser.add_argument("--streamlit", action="store_true", help="Start the Streamlit app")
    parser.add_argument("--api-key", type=str, help="Google API key for Gemini (can also use GOOGLE_API_KEY environment variable)")
    args = parser.parse_args()

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start agents
    for agent_config in AGENTS:
        # Add API key to AI Assistant Agent if provided
        if agent_config["name"] == "AI Assistant Agent" and args.api_key:
            agent_config["api_key"] = args.api_key

        process = start_agent(agent_config)
        processes.append(process)

        # Start a thread to log the output
        thread = threading.Thread(
            target=log_output,
            args=(process, agent_config["name"]),
            daemon=True
        )
        thread.start()

        # Wait a bit to avoid port conflicts
        time.sleep(1)

    # Start Streamlit if requested
    if args.streamlit:
        streamlit_process = start_streamlit()
        processes.append(streamlit_process)

        # Start a thread to log the output
        thread = threading.Thread(
            target=log_output,
            args=(streamlit_process, "Streamlit"),
            daemon=True
        )
        thread.start()

    logger.info("All agents started")

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
