import base64
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

FAKE_SECRET = "testsecret"
FAKE_BODY = b'{"events":[]}'


def make_signature(secret: str, body: bytes) -> str:
    h = hmac.new(secret.encode(), body, hashlib.sha256).digest()
    return base64.b64encode(h).decode()


def make_line_body(user_id: str, text: str, reply_token: str = "token123") -> bytes:
    return json.dumps({
        "events": [{
            "type": "message",
            "replyToken": reply_token,
            "source": {"userId": user_id},
            "message": {"type": "text", "text": text}
        }]
    }).encode()


def make_sig(body: bytes) -> str:
    h = hmac.new(FAKE_SECRET.encode(), body, hashlib.sha256).digest()
    return base64.b64encode(h).decode()


def test_cps_system_prompt_exists():
    from main import CPS_SYSTEM_PROMPT
    assert isinstance(CPS_SYSTEM_PROMPT, str)
    assert len(CPS_SYSTEM_PROMPT) > 200


def test_tools_definition():
    from main import TOOLS
    assert isinstance(TOOLS, list)
    assert len(TOOLS) == 1
    assert TOOLS[0]["name"] == "search_evidence"
    assert "question" in TOOLS[0]["input_schema"]["properties"]


def test_verify_signature_valid():
    from main import verify_signature
    sig = make_signature(FAKE_SECRET, FAKE_BODY)
    assert verify_signature(FAKE_BODY, sig, FAKE_SECRET) is True


def test_verify_signature_invalid():
    from main import verify_signature
    assert verify_signature(FAKE_BODY, "badsignature", FAKE_SECRET) is False


def test_add_message_and_trim():
    from main import add_message, history, MAX_HISTORY
    history.clear()
    user_id = "U123"
    for i in range(25):
        role = "user" if i % 2 == 0 else "assistant"
        add_message(user_id, role, f"message {i}")
    assert len(history[user_id]) == MAX_HISTORY


def test_history_order_preserved():
    from main import add_message, history
    history.clear()
    user_id = "U456"
    add_message(user_id, "user", "first")
    add_message(user_id, "assistant", "second")
    assert history[user_id][0]["content"] == "first"
    assert history[user_id][1]["content"] == "second"


def test_webhook_invalid_signature():
    from main import app
    client = TestClient(app)
    response = client.post("/webhook", content=b"{}", headers={"X-Line-Signature": "bad"})
    assert response.status_code == 401


def test_webhook_valid_returns_ok():
    from main import app, history
    history.clear()
    body = make_line_body("Uabc", "hello")
    sig = make_sig(body)

    with patch("main.LINE_CHANNEL_SECRET", FAKE_SECRET), \
         patch("main.call_claude", new=AsyncMock(return_value=("Hi there!", []))), \
         patch("main.send_reply", new=AsyncMock()):
        from main import app
        client = TestClient(app)
        response = client.post("/webhook", content=body, headers={"X-Line-Signature": sig})

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_oe_client_no_cookies():
    """ask() returns fallback string when OE_COOKIES_JSON is not set."""
    import asyncio
    import importlib
    import os
    os.environ.pop("OE_COOKIES_JSON", None)
    import main as m
    importlib.reload(m)
    result = asyncio.get_event_loop().run_until_complete(m.oe_client.ask("test"))
    assert result == "[文獻搜尋未設定，缺少 OE_COOKIES_JSON]"


def test_oe_client_extract_text_raw():
    from main import OpenEvidenceClient
    client = OpenEvidenceClient.__new__(OpenEvidenceClient)
    article = {"output": {"structured_article": {"raw_text": "hello evidence"}}}
    assert client._extract_text(article) == "hello evidence"


def test_oe_client_extract_text_fallback():
    from main import OpenEvidenceClient
    client = OpenEvidenceClient.__new__(OpenEvidenceClient)
    article = {"output": {"text": "fallback text"}}
    assert client._extract_text(article) == "fallback text"


def test_oe_client_extract_text_empty():
    from main import OpenEvidenceClient
    client = OpenEvidenceClient.__new__(OpenEvidenceClient)
    article = {"output": {}}
    assert client._extract_text(article) == "[文獻搜尋無結果]"


def test_split_message_short():
    from main import split_message
    result = split_message("hello")
    assert result == ["hello"]


def test_split_message_splits_on_paragraphs():
    from main import split_message
    para = "A" * 1000
    text = "\n\n".join([para] * 6)  # 6 paragraphs, total > 4800 chars
    result = split_message(text, limit=4800)
    assert len(result) > 1
    assert all(len(chunk) <= 4800 for chunk in result)


def test_split_message_caps_at_five():
    from main import split_message
    text = "\n\n".join(["B" * 1000] * 30)
    result = split_message(text, limit=4800)
    assert len(result) <= 5


def test_split_message_empty():
    from main import split_message
    assert split_message("") == [""]


def test_call_claude_no_tool_use():
    """call_claude returns text and unchanged history when no tool is triggered."""
    import asyncio
    from unittest.mock import MagicMock, patch

    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "診斷結果"

    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [mock_text_block]

    messages = [{"role": "user", "content": "咳嗽三天"}]

    with patch("main.claude.messages.create", return_value=mock_response):
        from main import call_claude
        text, updated = asyncio.get_event_loop().run_until_complete(
            call_claude(messages)
        )

    assert text == "診斷結果"
    assert updated == messages


def test_call_claude_with_tool_use():
    """call_claude executes tool, appends tool messages, returns second response text."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.id = "tu_123"
    mock_tool_block.name = "search_evidence"
    mock_tool_block.input = {"question": "S3 gallop LR for heart failure"}

    mock_resp1 = MagicMock()
    mock_resp1.stop_reason = "tool_use"
    mock_resp1.content = [mock_tool_block]

    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "根據文獻，最可能診斷為 CHF"

    mock_resp2 = MagicMock()
    mock_resp2.stop_reason = "end_turn"
    mock_resp2.content = [mock_text_block]

    messages = [{"role": "user", "content": "病患有 S3 gallop"}]

    with patch("main.claude.messages.create", side_effect=[mock_resp1, mock_resp2]), \
         patch("main.oe_client.ask", new=AsyncMock(return_value="S3 gallop LR+ 11.0 for CHF")):
        from main import call_claude
        text, updated = asyncio.get_event_loop().run_until_complete(
            call_claude(messages)
        )

    assert text == "根據文獻，最可能診斷為 CHF"
    assert len(updated) == 3
    assert updated[1]["role"] == "assistant"
    assert updated[2]["role"] == "user"
    assert updated[2]["content"][0]["type"] == "tool_result"
    assert updated[2]["content"][0]["content"] == "S3 gallop LR+ 11.0 for CHF"


def test_send_reply_multi_message():
    """send_reply sends all chunks as separate LINE messages."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    async def run():
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_post = AsyncMock(return_value=mock_response)

        with patch("main.LINE_CHANNEL_ACCESS_TOKEN", "tok"), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            from main import send_reply
            await send_reply("reply_token_abc", ["chunk1", "chunk2"])
            call_kwargs = mock_post.call_args
            messages_sent = call_kwargs.kwargs["json"]["messages"]
            assert len(messages_sent) == 2
            assert messages_sent[0]["text"] == "chunk1"
            assert messages_sent[1]["text"] == "chunk2"

    asyncio.get_event_loop().run_until_complete(run())
