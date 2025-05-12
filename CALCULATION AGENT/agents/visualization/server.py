import logging
import asyncio
import json
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

from agents.visualization.agent import visualization_agent, plot_positions


class VisualizationAgentTaskManager(InMemoryTaskManager):
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
        
        # Check if we have positions data
        positions_json = None
        for part in message.parts:
            if part.get("type") == "text":
                text = part.get("text", "")
                # Try to parse as JSON
                try:
                    json.loads(text)
                    positions_json = text
                    break
                except:
                    pass
            elif part.get("type") == "data":
                data = part.get("data", {})
                if "positions" in data:
                    positions_json = json.dumps(data["positions"])
                    break
        
        if not positions_json:
            return self._create_error_response(
                request, "Invalid request: No positions data provided"
            )
        
        # Call the plot_positions function
        result = plot_positions(positions_json)
        
        # Create response based on result
        if result.get("status") == "error":
            return self._create_error_response(
                request, result.get("error_message", "Unknown error")
            )
        
        # Update task with success response
        response_text = f"Successfully generated visualizations for {result.get('position_count', 0)} positions."
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
            
            # Add HTML as a file part if available
            if "html" in result:
                agent_response_parts.append({
                    "type": "file",
                    "file": {
                        "name": "visualization.html",
                        "mimeType": "text/html",
                        "bytes": result["html"]
                    }
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


def main(host="localhost", port=8004):
    # Define agent skills
    skill = AgentSkill(
        id="plot-positions",
        name="Plot Positions",
        description="Generates charts and tables for position data",
        tags=["visualization", "charts", "tables"],
        examples=["Generate charts for these positions"],
        inputModes=["text", "json"],
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
        name="Visualization Agent",
        description="Generates charts and tables for position data",
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
        defaultOutputModes=["text", "json", "html"],
        skills=[skill]
    )
    
    # Create task manager
    task_manager = VisualizationAgentTaskManager()
    
    # Create and start server
    server = A2AServer(
        agent_card=agent_card,
        task_manager=task_manager,
        host=host,
        port=port
    )
    
    logging.info(f"Starting Visualization Agent server on {host}:{port}")
    server.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
