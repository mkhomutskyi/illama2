"""Pydantic request/response schemas for the runner's HTTP API.

These mirror the OpenAI Chat / Completions request shape, with the multimodal
content extension that VLMPipeline needs (a content part can be either text
or ``image_url``). Kept in their own module so the schema surface is easy to
review in isolation and so unit tests can import it without dragging in the
full server (which depends on OpenVINO + Intel GPU runtime).

server.py re-imports these and exposes them at module scope so legacy
``import server; server.Message(...)`` style code keeps working.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ImageURL(BaseModel):
    url: str
    detail: Optional[str] = None


class ContentPartText(BaseModel):
    type: Literal["text"]
    text: str


class ContentPartImage(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Union[ContentPartText, ContentPartImage]


class Message(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = Field(default=False)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = Field(default=None)
    max_completion_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=0.7)
    top_p: Optional[float] = Field(default=1.0)
    presence_penalty: Optional[float] = Field(default=0.0)
    frequency_penalty: Optional[float] = Field(default=0.0)
    stop: Optional[List[str]] = None
    stream: Optional[bool] = Field(default=False)
    stream_options: Optional[StreamOptions] = None
    seed: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    # Tool calling: passed through to the tokenizer's chat template (so
    # Hermes-trained models like Qwen 2.5/3 see the tool schema in the
    # prompt) and parsed back out of the model's <tool_call>...</tool_call>
    # output. We do NOT enforce ``tool_choice`` server-side; the model picks
    # whether to call a tool, and we surface what it emits.
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = Field(default=None)
    max_completion_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=0.7)
    top_p: Optional[float] = Field(default=1.0)
    presence_penalty: Optional[float] = Field(default=0.0)
    frequency_penalty: Optional[float] = Field(default=0.0)
    stop: Optional[List[str]] = None
    stream: Optional[bool] = Field(default=False)
    stream_options: Optional[StreamOptions] = None
    seed: Optional[int] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "illama"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


__all__ = [
    "ImageURL",
    "ContentPartText",
    "ContentPartImage",
    "ContentPart",
    "Message",
    "StreamOptions",
    "ChatCompletionRequest",
    "CompletionRequest",
    "ModelInfo",
    "ModelsResponse",
]
