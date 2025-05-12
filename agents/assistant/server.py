import logging
import asyncio
import json
import argparse
import os
from pathlib import Path
from typing import AsyncIterable, Dict, Any

import google_a2a
from dotenv import load_dotenv
from google_a2a.common.server.task_manager import InMemoryTaskManager
from google_a2a.common.server import A2AServer
from google_a2a.common.types import (
    Artifact,
    JSONRPCResponse,
    Message,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    AgentSkill,
    AgentCapabilities,
    AgentCard
)
import google.generativeai as genai

from agents.assistant.agent import create_assistant_agent, analyze_data

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)


class AssistantAgentTaskManager(InMemoryTaskManager):
    def __init__(self):
        super().__init__()
        self.trades_data = {}
        self.positions_data = {}

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        # Upsert a task stored by InMemoryTaskManager
        await self.upsert_task(request.params)

        task_id = request.params.id

        # Extract message content
        message = request.params.message
        if message.role != "user" or not message.parts:
            return self._create_error_response(
                request, "Invalid request: Missing user message or parts"
            )

        # Extract question from message
        question = None
        trades_json = None
        positions_json = None

        for part in message.parts:
            if part.get("type") == "text":
                text = part.get("text", "")
                question = text
            elif part.get("type") == "data":
                data = part.get("data", {})
                if "trades" in data:
                    trades_json = json.dumps(data["trades"])
                    # Store trades data for future use
                    self.trades_data[task_id] = data["trades"]
                if "positions" in data:
                    positions_json = json.dumps(data["positions"])
                    # Store positions data for future use
                    self.positions_data[task_id] = data["positions"]

        # If no trades or positions data in the request, use stored data if available
        if not trades_json and task_id in self.trades_data:
            trades_json = json.dumps(self.trades_data[task_id])

        if not positions_json and task_id in self.positions_data:
            positions_json = json.dumps(self.positions_data[task_id])

        # Call the analyze_data function
        result = analyze_data(trades_json, positions_json, question)

        # Create response based on result
        if result.get("status") == "error":
            return self._create_error_response(
                request, result.get("error_message", "Unknown error")
            )

        # If the result contains a question and context, use the AI to generate an answer
        if "question" in result and "context" in result:
            # Format the context for the AI
            context_str = self._format_context_for_ai(result["context"])

            # Create a prompt for the AI
            prompt = f"""
            Question: {result["question"]}

            Context:
            {context_str}

            Please analyze the data and answer the question. Provide clear explanations and insights.
            """

            # Use the assistant agent to generate a response
            response = await self._generate_ai_response(prompt)

            # Update task with AI response
            task = await self._update_task(
                task_id=task_id,
                task_state=TaskState.COMPLETED,
                response_text=response
            )
        else:
            # Use the result's answer directly
            task = await self._update_task(
                task_id=task_id,
                task_state=TaskState.COMPLETED,
                response_text=result.get("answer", "I've analyzed your data.")
            )

        # Send the response
        return SendTaskResponse(id=request.id, result=task)

    async def on_send_task_subscribe(
        self,
        request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        # Not implemented for this agent
        return self._create_error_response(
            request, "Streaming not supported by this agent"
        )

    def _create_error_response(self, request, error_message):
        return JSONRPCResponse(
            id=request.id,
            error={
                "code": 400,
                "message": error_message
            }
        )

    async def _update_task(
        self,
        task_id: str,
        task_state: TaskState,
        response_text: str,
        result: Dict[str, Any] = None
    ) -> Task:
        # Get the task
        task = await self.get_task(task_id)

        # Update task status
        task.status = TaskStatus(
            state=task_state,
            message=Message(
                role="agent",
                parts=[{"type": "text", "text": response_text}]
            )
        )

        # Add result as artifact if provided
        if result:
            task.artifacts = [
                Artifact(
                    name="result",
                    parts=[{"type": "data", "data": result}]
                )
            ]

        # Update the task
        await self.upsert_task(task)

        return task

    def _format_context_for_ai(self, context: Dict[str, Any]) -> str:
        """Format the context data for the AI to process."""
        context_str = ""

        # Add trade information if available
        if "trades" in context:
            trades = context["trades"]
            context_str += f"Trade Data: {len(trades)} trades\n"

            # Add sample of trades (limit to 5 for brevity)
            if trades:
                context_str += "Sample trades:\n"
                for i, trade in enumerate(trades[:5]):
                    context_str += f"Trade {i+1}: {json.dumps(trade)}\n"

                if len(trades) > 5:
                    context_str += f"... and {len(trades) - 5} more trades\n"

        # Add position information if available
        if "positions" in context:
            positions = context["positions"]
            context_str += f"\nPosition Data: {len(positions)} positions\n"

            # Add all positions (usually there aren't too many)
            if positions:
                context_str += "Positions:\n"
                for i, position in enumerate(positions):
                    context_str += f"Position {i+1}: {json.dumps(position)}\n"

        return context_str

    async def _generate_ai_response(self, prompt: str) -> str:
        """Use the Gemini model to generate a response."""
        try:
            # Use the Gemini model directly
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)

            # Extract and return the text response
            if response and response.text:
                return response.text
            else:
                return "I've analyzed your data but couldn't generate a detailed response."
        except Exception as e:
            logging.error(f"Error generating AI response: {str(e)}")
            return f"I'm sorry, I couldn't analyze your data at this time. Error: {str(e)}"


def main(host="localhost", port=8005, api_key=None):
    # Initialize the Gemini model with API key
    try:
        # Use provided API key or get from environment
        google_api_key = api_key or os.environ.get("GOOGLE_API_KEY")

        if not google_api_key:
            raise ValueError(
                "Google API key is required. Either pass it as an argument or "
                "set the GOOGLE_API_KEY environment variable."
            )

        # Configure the Google Generative AI library with the API key
        genai.configure(api_key=google_api_key)
        logging.info("Successfully configured Gemini API with provided key")
    except Exception as e:
        logging.error(f"Failed to initialize Gemini API: {str(e)}")
        raise

    # Define agent skills
    skill = AgentSkill(
        id="analyze-trade-data",
        name="Analyze Trade Data",
        description="Analyzes trade and position data to answer questions",
        tags=["analysis", "AI", "assistant"],
        examples=["What are my largest positions?", "How many trades do I have for GOLD?"],
        inputModes=["text", "json"],
        outputModes=["text"]
    )

    # Define agent capabilities
    capabilities = AgentCapabilities(
        streaming=False,
        pushNotifications=False,
        stateTransitionHistory=False
    )

    # Create agent card
    agent_card = AgentCard(
        name="AI Assistant",
        description="AI assistant that analyzes trade data and answers questions",
        url=f"http://{host}:{port}",
        provider={
            "organization": "Calculation Agent",
            "url": "https://example.com"
        },
        version="1.0.0",
        authentication={
            "schemes": ["None"]
        },
        capabilities=capabilities,
        defaultInputModes=["text", "json"],
        defaultOutputModes=["text"],
        skills=[skill]
    )

    # Create task manager
    task_manager = AssistantAgentTaskManager()

    # Create and start server
    server = A2AServer(
        agent_card=agent_card,
        task_manager=task_manager,
        host=host,
        port=port
    )

    logging.info(f"Starting AI Assistant Agent server on {host}:{port}")
    server.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start the AI Assistant Agent")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8005, help="Port to bind to")
    parser.add_argument("--api-key", type=str, help="Google API key for Gemini (can also use GOOGLE_API_KEY environment variable)")
    args = parser.parse_args()

    main(args.host, args.port, args.api_key)
