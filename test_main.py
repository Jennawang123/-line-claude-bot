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

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hi there!")]

    with patch("main.LINE_CHANNEL_SECRET", FAKE_SECRET), \
         patch("main.claude.messages.create", return_value=mock_response), \
         patch("main.send_reply", new_callable=AsyncMock):
        from main import app
        client = TestClient(app)
        response = client.post("/webhook", content=body, headers={"X-Line-Signature": sig})

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
