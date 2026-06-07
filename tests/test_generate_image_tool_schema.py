"""Regression coverage for exposing image generation to native tool callers."""
import json

from src.agent_tools import TOOL_TAGS
from src.tool_index import BUILTIN_TOOL_DESCRIPTIONS, ToolIndex
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS, function_call_to_tool_block


def test_generate_image_has_native_function_schema():
    names = {s["function"]["name"] for s in FUNCTION_TOOL_SCHEMAS}
    assert "generate_image" in names


def test_generate_image_function_call_serializes_to_tool_block():
    args = {
        "prompt": "a red fox in snow",
        "model": "qwen-image-bf16",
        "size": "1024x1024",
        "quality": "low",
    }

    block = function_call_to_tool_block("generate_image", json.dumps(args))

    assert block is not None
    assert block.tool_type == "generate_image"
    assert block.content == "a red fox in snow\nqwen-image-bf16\n1024x1024\nlow"


def test_generate_image_function_call_omits_optional_blank_lines_at_end():
    block = function_call_to_tool_block(
        "generate_image",
        json.dumps({"prompt": "a tiny spaceship"}),
    )

    assert block is not None
    assert block.tool_type == "generate_image"
    assert block.content == "a tiny spaceship"


def test_generate_image_registered_for_prompt_and_retrieval():
    assert "generate_image" in TOOL_TAGS
    assert "generate_image" in BUILTIN_TOOL_DESCRIPTIONS
    assert any(
        "generate_image" in tools and "generate image" in keywords
        for keywords, tools in ToolIndex._KEYWORD_HINTS.items()
    )
