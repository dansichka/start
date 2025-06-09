from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Type
import inspect

from fastapi import APIRouter
from pydantic import BaseModel


@dataclass
class Tool:
    """Description of a single tool exposed via the MCP server."""

    name: str
    description: str
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    function: Callable


class MCPServer:
    """Minimal MCP server to expose tools via FastAPI routes."""

    def __init__(self, name: str, description: str, tools: List[Tool]):
        self.name = name
        self.description = description
        self.tools = tools
        self.router = APIRouter(prefix=f"/{name}")
        for tool in tools:
            self.router.post(
                f"/{tool.name}",
                response_model=tool.output_model,
                name=tool.name,
                summary=tool.description,
            )(self._create_handler(tool))

    def _create_handler(self, tool: Tool):
        async def handler(data: tool.input_model):
            if inspect.iscoroutinefunction(tool.function):
                result = await tool.function(**data.model_dump())
            else:
                result = tool.function(**data.model_dump())
            return result

        return handler
