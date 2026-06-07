"""Regression coverage for the chat-agent generated-image edit tool."""
import asyncio
import base64
import json
from types import SimpleNamespace

from core.models import ChatMessage
from src.agent_tools import TOOL_TAGS
from src.settings import DEFAULT_SETTINGS
from src.tool_index import BUILTIN_TOOL_DESCRIPTIONS, ToolIndex
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS, function_call_to_tool_block


def test_edit_generated_image_has_native_function_schema():
    names = {s["function"]["name"] for s in FUNCTION_TOOL_SCHEMAS}
    assert "edit_generated_image" in names


def test_edit_generated_image_function_call_serializes_to_json_tool_block():
    args = {
        "prompt": "make the jacket red but keep the pose",
        "image_id": "img123",
        "model": "qwen-image-edit-bf16",
        "size": "1024x1024",
        "quality": "low",
    }

    block = function_call_to_tool_block("edit_generated_image", json.dumps(args))

    assert block is not None
    assert block.tool_type == "edit_generated_image"
    assert json.loads(block.content) == args


def test_edit_generated_image_registered_for_prompt_and_retrieval():
    assert "edit_generated_image" in TOOL_TAGS
    assert "edit_generated_image" in BUILTIN_TOOL_DESCRIPTIONS
    assert any(
        "edit_generated_image" in tools and "edit image" in keywords
        for keywords, tools in ToolIndex._KEYWORD_HINTS.items()
    )
    assert any(
        "edit_generated_image" in tools and "attached image" in keywords
        for keywords, tools in ToolIndex._KEYWORD_HINTS.items()
    )


def test_image_edit_model_setting_exists():
    assert "image_edit_model" in DEFAULT_SETTINGS


class _Expr:
    def __eq__(self, _other):
        return self

    def ilike(self, _pattern):
        return self

    def desc(self):
        return self

    def __or__(self, _other):
        return self


class _FakeGalleryImage:
    id = _Expr()
    owner = _Expr()
    is_active = _Expr()
    filename = _Expr()
    updated_at = _Expr()
    created_at = _Expr()
    session_id = _Expr()

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _EmptyQuery:
    def filter(self, *_args, **_kwargs):
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def first(self):
        return None


class _FakeDb:
    def __init__(self):
        self.added = []

    def query(self, *_args, **_kwargs):
        return _EmptyQuery()

    def add(self, item):
        self.added.append(item)

    def commit(self):
        pass

    def close(self):
        pass


class _FakeEditResponse:
    status_code = 200

    def json(self):
        return {"data": [{"b64_json": base64.b64encode(b"edited-image").decode("ascii")}]}


class _FakeAsyncClient:
    posted = None
    timeout = None

    def __init__(self, *_args, **kwargs):
        type(self).timeout = kwargs.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def post(self, url, data=None, files=None, headers=None):
        type(self).posted = {"url": url, "data": data, "files": files, "headers": headers}
        return _FakeEditResponse()


def test_edit_generated_image_falls_back_to_latest_attached_image(monkeypatch, tmp_path):
    import httpx
    import src.ai_interaction as ai_interaction
    import src.constants as constants
    import src.database as database

    upload_id = "a" * 32 + ".png"
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    source_path = upload_dir / upload_id
    source_path.write_bytes(b"source-image")
    (upload_dir / "uploads.json").write_text(
        json.dumps({
            "alice:hash": {
                "id": upload_id,
                "path": str(source_path),
                "mime": "image/png",
                "size": 12,
                "name": "poster.png",
                "original_name": "poster.png",
                "owner": "alice",
                "width": 640,
                "height": 480,
            }
        }),
        encoding="utf-8",
    )

    sess = SimpleNamespace(history=[
        ChatMessage(
            "user",
            "change the title",
            metadata={"attachments": [{"id": upload_id, "name": "poster.png", "mime": "image/png"}]},
        )
    ])
    fake_db = _FakeDb()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(constants, "BASE_DIR", str(tmp_path) + "/")
    monkeypatch.setattr(constants, "UPLOAD_DIR", str(upload_dir))
    monkeypatch.setattr(database, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(database, "GalleryImage", _FakeGalleryImage, raising=False)
    monkeypatch.setattr(
        ai_interaction,
        "_session_manager",
        SimpleNamespace(get_session=lambda _sid: sess),
    )
    monkeypatch.setattr(
        ai_interaction,
        "_resolve_model",
        lambda *_args, **_kwargs: ("http://localhost:8000/v1/chat/completions", "qwen-image-edit-bf16", {}),
    )
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    result = asyncio.run(ai_interaction.do_edit_generated_image(
        json.dumps({
            "prompt": "change the title to Launch Day",
            "model": "qwen-image-edit-bf16",
        }),
        session_id="sid",
        owner="alice",
    ))

    assert result["source_image_id"] == upload_id
    assert result["image_size"] == "640x480"
    assert result["image_url"].startswith("/api/generated-image/")
    assert _FakeAsyncClient.timeout.read == 900.0
    assert _FakeAsyncClient.posted["url"] == "http://localhost:8000/v1/images/edits"
    assert _FakeAsyncClient.posted["data"]["prompt"] == "change the title to Launch Day"
    assert _FakeAsyncClient.posted["files"]["image"][0] == "poster.png"
    assert _FakeAsyncClient.posted["files"]["image"][1] == b"source-image"


def test_edit_generated_image_prefers_latest_generated_tool_event(monkeypatch, tmp_path):
    import httpx
    import src.ai_interaction as ai_interaction
    import src.database as database

    generated_dir = tmp_path / "data" / "generated_images"
    generated_dir.mkdir(parents=True)
    taco_filename = "aaaaaaaaaaaa.png"
    mac_filename = "bbbbbbbbbbbb.png"
    (generated_dir / taco_filename).write_bytes(b"taco-image")
    (generated_dir / mac_filename).write_bytes(b"mac-image")

    sess = SimpleNamespace(history=[
        ChatMessage(
            "assistant",
            "Generated image for: tacos",
            metadata={"tool_events": [{
                "tool": "generate_image",
                "image_url": f"/api/generated-image/{taco_filename}",
                "image_prompt": "tacos",
            }]},
        ),
        ChatMessage(
            "assistant",
            "Generated image for: mac n cheese",
            metadata={"tool_events": [{
                "tool": "generate_image",
                "image_url": f"/api/generated-image/{mac_filename}",
                "image_prompt": "mac n cheese",
            }]},
        ),
    ])
    fake_db = _FakeDb()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(database, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(database, "GalleryImage", _FakeGalleryImage, raising=False)
    monkeypatch.setattr(
        ai_interaction,
        "_session_manager",
        SimpleNamespace(get_session=lambda _sid: sess),
    )
    monkeypatch.setattr(
        ai_interaction,
        "_resolve_model",
        lambda *_args, **_kwargs: ("http://localhost:8000/v1/chat/completions", "qwen-image-edit-bf16", {}),
    )
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    result = asyncio.run(ai_interaction.do_edit_generated_image(
        json.dumps({
            "prompt": "add the words Mac N Cheese above the food",
            "model": "qwen-image-edit-bf16",
        }),
        session_id="sid",
        owner="alice",
    ))

    assert result["source_image_id"] == mac_filename
    assert _FakeAsyncClient.posted["files"]["image"][0] == mac_filename
    assert _FakeAsyncClient.posted["files"]["image"][1] == b"mac-image"
