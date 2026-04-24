import base64
import hashlib
import hmac
import json
import os
from collections import defaultdict

import anthropic
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI()

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

history: dict[str, list[dict]] = defaultdict(list)
MAX_HISTORY = 20


def verify_signature(body: bytes, signature: str, secret: str = None) -> bool:
    if secret is None:
        secret = LINE_CHANNEL_SECRET
    h = hmac.new(secret.encode(), body, hashlib.sha256).digest()
    expected = base64.b64encode(h).decode()
    return hmac.compare_digest(expected, signature)


def add_message(user_id: str, role: str, content: str) -> None:
    history[user_id].append({"role": role, "content": content})
    if len(history[user_id]) > MAX_HISTORY:
        history[user_id] = history[user_id][-MAX_HISTORY:]


async def send_reply(reply_token: str, message: str) -> None:
    async with httpx.AsyncClient() as http:
        await http.post(
            "https://api.line.me/v2/bot/message/reply",
            headers={
                "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "replyToken": reply_token,
                "messages": [{"type": "text", "text": message}],
            },
            timeout=10.0,
        )


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()

    if not verify_signature(body, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    data = json.loads(body)

    for event in data.get("events", []):
        if event.get("type") != "message":
            continue
        if event.get("message", {}).get("type") != "text":
            continue

        user_id = event["source"]["userId"]
        reply_token = event["replyToken"]
        user_text = event["message"]["text"]

        add_message(user_id, "user", user_text)

        response = claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=history[user_id],
        )

        assistant_text = response.content[0].text
        add_message(user_id, "assistant", assistant_text)

        await send_reply(reply_token, assistant_text)

    return JSONResponse(content={"status": "ok"})
