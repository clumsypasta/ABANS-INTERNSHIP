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

from agents.pricing.agent import pricing_agent, get_price


class PricingAgentTaskManager(InMemoryTaskManager):
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
        
        # Extract symbol and exchange from message
        symbol = None
        exchange = None
        
        for part in message.parts:
            if part.get("type") == "text":
                text = part.get("text", "")
                # Simple parsing for demo purposes
                # In a real implementation, use NLP or more robust parsing
                if "symbol=" in text or "symbol:" in text:
                    for segment in text.split():
                        if segment.startswith("symbol=") or segment.startswith("symbol:"):
                            symbol = segment.split("=")[1] if "=" in segment else segment.split(":")[1]
                if "exchange=" in text or "exchange:" in text:
                    for segment in text.split():
                        if segment.startswith("exchange=") or segment.startswith("exchange:"):
                            exchange = segment.split("=")[1] if "=" in segment else segment.split(":")[1]
            elif part.get("type") == "data":
                data = part.get("data", {})
                symbol = data.get("symbol")
                exchange = data.get("exchange")
        
        if not symbol:
            return self._create_error_response(
                request, "Invalid request: No symbol provided"
            )
        
        # Call the get_price function
        result = get_price(symbol, exchange)
        
        # Create response based on result
        if result.get("status") == "error":
            return self._create_error_response(
                request, result.get("error_message", "Unknown error")
            )
        
        # Update task with success response
        price = result.get("price", 0)
        currency = result.get("currency", "USD")
        response_text = f"Current price for {symbol}: {price} {currency}"
        
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


def main(host="localhost", port=8003):
    # Define agent skills
    skill = AgentSkill(
        id="get-price",
        name="Get Price",
        description="Fetches real-time prices for commodities and other financial instruments",
        tags=["pricing", "market", "commodities"],
        examples=["Get the current price of GOLD", "What's the price of SILVER on MCX?"],
        inputModes=["text"],
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
        name="Pricing Agent",
        description="Fetches real-time prices for commodities and other financial instruments",
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
        defaultInputModes=["text"],
        defaultOutputModes=["text", "json"],
        skills=[skill]
    )
    
    # Create task manager
    task_manager = PricingAgentTaskManager()
    
    # Create and start server
    server = A2AServer(
        agent_card=agent_card,
        task_manager=task_manager,
        host=host,
        port=port
    )
    
    logging.info(f"Starting Pricing Agent server on {host}:{port}")
    server.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
