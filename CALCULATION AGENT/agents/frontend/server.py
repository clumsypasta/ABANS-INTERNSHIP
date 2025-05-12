import logging
import asyncio
import json
import base64
import os
from typing import AsyncIterable, Dict, Any
import uuid

import google_a2a
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
from google_a2a.common.client import A2AClient

# Import Streamlit app
from agents.frontend.app import app


class FrontendAgentTaskManager(InMemoryTaskManager):
    def __init__(self):
        super().__init__()

        # Initialize A2A clients for other agents
        self.data_agent_client = A2AClient("http://localhost:8001")
        self.compute_agent_client = A2AClient("http://localhost:8002")
        self.pricing_agent_client = A2AClient("http://localhost:8003")
        self.visualization_agent_client = A2AClient("http://localhost:8004")
        self.assistant_agent_client = A2AClient("http://localhost:8005")

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

        # Check if we have a command
        command = None
        file_part = None

        for part in message.parts:
            if part.get("type") == "text":
                text = part.get("text", "")
                if text.startswith("/"):
                    command = text[1:].strip()
            elif part.get("type") == "file":
                file_part = part

        # Process based on command or file
        if command == "help":
            # Return help information
            response_text = """
            Available commands:
            /help - Show this help message
            /parse - Parse a trade data file
            /compute - Compute positions from trades
            /price <symbol> - Get current price for a symbol
            /visualize - Generate visualizations for positions
            /ask <question> - Ask the AI Assistant a question about your data

            You can also upload a file to parse it.
            """
            task = await self._update_task(
                task_id=task_id,
                task_state=TaskState.COMPLETED,
                response_text=response_text
            )
            return SendTaskResponse(id=request.id, result=task)

        elif command and command.startswith("parse"):
            # Check if we have a file
            if not file_part:
                return self._create_error_response(
                    request, "No file provided for parsing"
                )

            # Call Data Agent
            data_agent_response = await self._call_data_agent(file_part)

            if data_agent_response.get("status") == "error":
                return self._create_error_response(
                    request, data_agent_response.get("error_message", "Error parsing file")
                )

            # Update task with success response
            response_text = f"Successfully parsed {data_agent_response.get('count', 0)} trades."
            task = await self._update_task(
                task_id=task_id,
                task_state=TaskState.COMPLETED,
                response_text=response_text,
                result=data_agent_response
            )
            return SendTaskResponse(id=request.id, result=task)

        elif command and command.startswith("compute"):
            # Check if we have trades data in the state
            trades_data = self._get_trades_from_state(task_id)
            if not trades_data:
                return self._create_error_response(
                    request, "No trades data available. Please parse a file first."
                )

            # Call Compute Agent
            compute_agent_response = await self._call_compute_agent(trades_data)

            if compute_agent_response.get("status") == "error":
                return self._create_error_response(
                    request, compute_agent_response.get("error_message", "Error computing positions")
                )

            # Update task with success response
            response_text = f"Successfully computed {compute_agent_response.get('count', 0)} positions."
            task = await self._update_task(
                task_id=task_id,
                task_state=TaskState.COMPLETED,
                response_text=response_text,
                result=compute_agent_response
            )
            return SendTaskResponse(id=request.id, result=task)

        elif command and command.startswith("price"):
            # Extract symbol from command
            parts = command.split()
            if len(parts) < 2:
                return self._create_error_response(
                    request, "Please specify a symbol (e.g., /price GOLD)"
                )

            symbol = parts[1]

            # Call Pricing Agent
            pricing_agent_response = await self._call_pricing_agent(symbol)

            if pricing_agent_response.get("status") == "error":
                return self._create_error_response(
                    request, pricing_agent_response.get("error_message", "Error fetching price")
                )

            # Update task with success response
            price = pricing_agent_response.get("price", 0)
            currency = pricing_agent_response.get("currency", "USD")
            response_text = f"Current price for {symbol}: {price} {currency}"

            task = await self._update_task(
                task_id=task_id,
                task_state=TaskState.COMPLETED,
                response_text=response_text,
                result=pricing_agent_response
            )
            return SendTaskResponse(id=request.id, result=task)

        elif command and command.startswith("visualize"):
            # Check if we have positions data in the state
            positions_data = self._get_positions_from_state(task_id)
            if not positions_data:
                return self._create_error_response(
                    request, "No positions data available. Please compute positions first."
                )

            # Call Visualization Agent
            visualization_agent_response = await self._call_visualization_agent(positions_data)

            if visualization_agent_response.get("status") == "error":
                return self._create_error_response(
                    request, visualization_agent_response.get("error_message", "Error generating visualizations")
                )

            # Update task with success response
            response_text = f"Successfully generated visualizations for {visualization_agent_response.get('position_count', 0)} positions."
            task = await self._update_task(
                task_id=task_id,
                task_state=TaskState.COMPLETED,
                response_text=response_text,
                result=visualization_agent_response
            )
            return SendTaskResponse(id=request.id, result=task)

        elif command and command.startswith("ask"):
            # Extract question from command
            question = command[4:].strip()
            if not question:
                return self._create_error_response(
                    request, "Please provide a question after /ask"
                )

            # Get trades and positions data from state
            trades_data = self._get_trades_from_state(task_id)
            positions_data = self._get_positions_from_state(task_id)

            if not trades_data and not positions_data:
                return self._create_error_response(
                    request, "No data available. Please upload and process trade data first."
                )

            # Call AI Assistant Agent
            response_text = await self._call_assistant_agent(question, trades_data, positions_data)

            # Update task with AI response
            task = await self._update_task(
                task_id=task_id,
                task_state=TaskState.COMPLETED,
                response_text=response_text
            )
            return SendTaskResponse(id=request.id, result=task)

        elif file_part:
            # Automatically parse the file
            data_agent_response = await self._call_data_agent(file_part)

            if data_agent_response.get("status") == "error":
                return self._create_error_response(
                    request, data_agent_response.get("error_message", "Error parsing file")
                )

            # Update task with success response
            response_text = f"Successfully parsed {data_agent_response.get('count', 0)} trades."
            task = await self._update_task(
                task_id=task_id,
                task_state=TaskState.COMPLETED,
                response_text=response_text,
                result=data_agent_response
            )
            return SendTaskResponse(id=request.id, result=task)

        else:
            # Default response
            response_text = "Welcome to the Trade Position Calculator. You can upload a file or use commands like /help, /parse, /compute, /price, or /visualize."
            task = await self._update_task(
                task_id=task_id,
                task_state=TaskState.COMPLETED,
                response_text=response_text
            )
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
        result: dict = None
    ) -> Task:
        task = self.tasks[task_id]

        # Create response parts
        agent_response_parts = [
            {
                "type": "text",
                "text": response_text,
            }
        ]

        # Add result data if available
        if result:
            agent_response_parts.append({
                "type": "data",
                "data": result
            })

        # Update task status
        task.status = TaskStatus(
            state=task_state,
            message=Message(
                role="agent",
                parts=agent_response_parts,
            )
        )

        # Add artifacts
        task.artifacts = [
            Artifact(
                parts=agent_response_parts,
                index=0
            )
        ]

        return task

    async def _call_data_agent(self, file_part):
        """Call the Data Agent to parse a file."""
        try:
            # Extract file data
            file_data = file_part.get("file", {})
            file_bytes = file_data.get("bytes", "")
            mime_type = file_data.get("mimeType", "")

            if not file_bytes or not mime_type:
                return {
                    "status": "error",
                    "error_message": "Invalid file data"
                }

            # Create task ID
            task_id = str(uuid.uuid4())

            # Create message
            message = {
                "role": "user",
                "parts": [
                    {
                        "type": "file",
                        "file": {
                            "bytes": file_bytes,
                            "mimeType": mime_type
                        }
                    }
                ]
            }

            # Send task to Data Agent
            response = await self.data_agent_client.send_task({
                "id": task_id,
                "message": message
            })

            # Extract result from response
            if response and response.artifacts and response.artifacts[0].parts:
                for part in response.artifacts[0].parts:
                    if part.get("type") == "data":
                        return part.get("data", {})

            return {
                "status": "error",
                "error_message": "Failed to parse file"
            }

        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Error calling Data Agent: {str(e)}"
            }

    async def _call_compute_agent(self, trades_data):
        """Call the Compute Agent to compute positions."""
        try:
            # Create task ID
            task_id = str(uuid.uuid4())

            # Create message
            message = {
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "text": json.dumps(trades_data)
                    }
                ]
            }

            # Send task to Compute Agent
            response = await self.compute_agent_client.send_task({
                "id": task_id,
                "message": message
            })

            # Extract result from response
            if response and response.artifacts and response.artifacts[0].parts:
                for part in response.artifacts[0].parts:
                    if part.get("type") == "data":
                        return part.get("data", {})

            return {
                "status": "error",
                "error_message": "Failed to compute positions"
            }

        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Error calling Compute Agent: {str(e)}"
            }

    async def _call_pricing_agent(self, symbol, exchange=None):
        """Call the Pricing Agent to get a price."""
        try:
            # Create task ID
            task_id = str(uuid.uuid4())

            # Create message
            text = f"symbol={symbol}"
            if exchange:
                text += f" exchange={exchange}"

            message = {
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }

            # Send task to Pricing Agent
            response = await self.pricing_agent_client.send_task({
                "id": task_id,
                "message": message
            })

            # Extract result from response
            if response and response.artifacts and response.artifacts[0].parts:
                for part in response.artifacts[0].parts:
                    if part.get("type") == "data":
                        return part.get("data", {})

            return {
                "status": "error",
                "error_message": "Failed to get price"
            }

        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Error calling Pricing Agent: {str(e)}"
            }

    async def _call_visualization_agent(self, positions_data):
        """Call the Visualization Agent to generate visualizations."""
        try:
            # Create task ID
            task_id = str(uuid.uuid4())

            # Create message
            message = {
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "text": json.dumps(positions_data)
                    }
                ]
            }

            # Send task to Visualization Agent
            response = await self.visualization_agent_client.send_task({
                "id": task_id,
                "message": message
            })

            # Extract result from response
            if response and response.artifacts and response.artifacts[0].parts:
                for part in response.artifacts[0].parts:
                    if part.get("type") == "data":
                        return part.get("data", {})

            return {
                "status": "error",
                "error_message": "Failed to generate visualizations"
            }

        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Error calling Visualization Agent: {str(e)}"
            }

    def _get_trades_from_state(self, task_id):
        """Get trades data from task state."""
        task = self.tasks.get(task_id)
        if not task or not task.artifacts:
            return None

        for artifact in task.artifacts:
            for part in artifact.parts:
                if part.get("type") == "data" and "trades" in part.get("data", {}):
                    return part.get("data", {}).get("trades")

        return None

    def _get_positions_from_state(self, task_id):
        """Get positions data from task state."""
        task = self.tasks.get(task_id)
        if not task or not task.artifacts:
            return None

        for artifact in task.artifacts:
            for part in artifact.parts:
                if part.get("type") == "data" and "positions" in part.get("data", {}):
                    return part.get("data", {}).get("positions")

        return None

    async def _call_assistant_agent(self, question, trades_data=None, positions_data=None):
        """Call the AI Assistant Agent to answer questions about the data."""
        try:
            # Create task ID
            task_id = str(uuid.uuid4())

            # Create message parts
            parts = [
                {
                    "type": "text",
                    "text": question
                }
            ]

            # Add data if available
            if trades_data or positions_data:
                data = {}
                if trades_data:
                    data["trades"] = trades_data
                if positions_data:
                    data["positions"] = positions_data

                parts.append({
                    "type": "data",
                    "data": data
                })

            # Create message
            message = {
                "role": "user",
                "parts": parts
            }

            # Send task to Assistant Agent
            response = await self.assistant_agent_client.send_task({
                "id": task_id,
                "message": message
            })

            # Extract response text
            if response and response.status and response.status.message:
                for part in response.status.message.parts:
                    if part.get("type") == "text":
                        return part.get("text", "")

            return "I couldn't analyze your data at this time."

        except Exception as e:
            logging.error(f"Error calling Assistant Agent: {str(e)}")
            return f"Error analyzing data: {str(e)}"


def main(host="localhost", port=8000):
    # Define agent skills
    skill = AgentSkill(
        id="trade-calculator",
        name="Trade Calculator",
        description="Calculates FIFO and LIFO weighted average prices for open positions",
        tags=["trades", "positions", "FIFO", "LIFO"],
        examples=["Upload a trade file", "/compute", "/price GOLD"],
        inputModes=["text", "file"],
        outputModes=["text", "json", "html"]
    )

    # Define agent capabilities
    capabilities = AgentCapabilities(
        streaming=False,
        pushNotifications=False,
        stateTransitionHistory=False
    )

    # Create agent card
    agent_card = AgentCard(
        name="Frontend Agent",
        description="Provides a user interface for the Trade Position Calculator",
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
        defaultInputModes=["text", "file"],
        defaultOutputModes=["text", "json", "html"],
        skills=[skill]
    )

    # Create task manager
    task_manager = FrontendAgentTaskManager()

    # Create and start server
    server = A2AServer(
        agent_card=agent_card,
        task_manager=task_manager,
        host=host,
        port=port
    )

    logging.info(f"Starting Frontend Agent server on {host}:{port}")
    server.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
