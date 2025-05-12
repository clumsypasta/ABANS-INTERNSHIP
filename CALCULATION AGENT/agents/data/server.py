import logging
import asyncio
from typing import AsyncIterable

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

from agents.data.agent import data_agent, parse_file


class DataAgentTaskManager(InMemoryTaskManager):
    def __init__(self):
        super().__init__()

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
        
        # Check if we have file data
        file_part = None
        for part in message.parts:
            if part.get("type") == "file" and part.get("file"):
                file_part = part
                break
        
        if not file_part:
            return self._create_error_response(
                request, "Invalid request: No file provided"
            )
        
        # Extract file data
        file_data = file_part.get("file", {})
        file_bytes = file_data.get("bytes", "")
        mime_type = file_data.get("mimeType", "")
        
        if not file_bytes or not mime_type:
            return self._create_error_response(
                request, "Invalid request: Missing file bytes or MIME type"
            )
        
        # Call the parse_file function
        result = parse_file(file_bytes, mime_type)
        
        # Create response based on result
        if result.get("status") == "error":
            return self._create_error_response(
                request, result.get("error_message", "Unknown error")
            )
        
        # Update task with success response
        response_text = f"Successfully parsed {result.get('count', 0)} trades."
        task = await self._update_task(
            task_id=task_id,
            task_state=TaskState.COMPLETED,
            response_text=response_text,
            result=result
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


def main(host="localhost", port=8001):
    # Define agent skills
    skill = AgentSkill(
        id="parse-trade-data",
        name="Parse Trade Data",
        description="Parses CSV/XLS(X) files into a uniform JSON trade list",
        tags=["data", "parsing", "trades"],
        examples=["Parse this trade data file"],
        inputModes=["text", "file"],
        outputModes=["text", "json"]
    )
    
    # Define agent capabilities
    capabilities = AgentCapabilities(
        streaming=False,
        pushNotifications=False,
        stateTransitionHistory=False
    )
    
    # Create agent card
    agent_card = AgentCard(
        name="Data Agent",
        description="Parses CSV/XLS(X) files into a uniform JSON trade list",
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
        defaultOutputModes=["text", "json"],
        skills=[skill]
    )
    
    # Create task manager
    task_manager = DataAgentTaskManager()
    
    # Create and start server
    server = A2AServer(
        agent_card=agent_card,
        task_manager=task_manager,
        host=host,
        port=port
    )
    
    logging.info(f"Starting Data Agent server on {host}:{port}")
    server.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
