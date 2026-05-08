import asyncio
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

CPS_SYSTEM_PROMPT = """你是一位內科主治醫師，使用 NEJM Clinical Problem-Solving 格式進行臨床推理。

每次收到病例或臨床問題時，依照以下步驟推理：

## 推理步驟

**步驟 1 — 問題重述**
一句話總結：「這是一位 [年齡][性別]，有 [重要病史]，以 [主訴持續時間] 的 [主訴] 就診，伴隨 [關鍵症狀]。」

**步驟 2 — 鑑別診斷（Top 5–8）**
依預測機率（%）排序。每個診斷標註：
- 預測機率
- 支持證據（來自病例）
- 反對證據
- ⚠️ 標記不可漏診（高死亡率/高致殘率）

必須考慮的不可漏診類別：
- 心血管：ACS、主動脈剝離、肺栓塞、心包填塞
- 神經：蜘蛛膜下腔出血、腦中風、腦膜炎、硬膜外血腫
- 外科急症：AAA破裂、腸穿孔、子宮外孕
- 代謝：敗血症、腎上腺危象、甲狀腺風暴

**步驟 3 — 關鍵鑑別點**
針對最可能的 2–3 個診斷，列出能推高或降低機率的關鍵發現，附 LR+/LR−（已知時）。

LR 解讀：
- LR+ > 10 或 LR− < 0.1：強力改變機率
- LR+ 5–10 或 LR− 0.1–0.2：中度改變
- LR+ 2–5 或 LR− 0.2–0.5：輕度改變

**步驟 4 — 文獻查詢（工具呼叫）**
以下情況呼叫 search_evidence 工具：
- 罕見診斷需要最新指引
- 兩個診斷機率相近，文獻有助區分
- 需要具體治療方案或藥物劑量
- 搜尋問題請用英文以獲得最佳結果

**步驟 5 — Bayesian 更新**（若有呼叫 search_evidence）
使用文獻資料重新計算後驗機率：
後驗勝算 = 先驗勝算 × LR
後驗機率 = 後驗勝算 / (1 + 後驗勝算)

**步驟 6 — 結論**
- 最可能診斷（附最終機率）
- 建議檢查（依優先順序）
- 緊急處置（若有急症）

## 輸出規範
- 語言：繁體中文
- 格式：清楚分節標題 + 條列式（bullet points）
- 禁止使用 Markdown 表格（| 欄位 | 格式 |），改用條列式取代
- 禁止使用程式碼區塊（``` 包覆的 ASCII 圖）
- 包含 LR+/LR− 數值（已知時）
- 無論機率高低，⚠️ 不可漏診診斷必須列出
- 若問題非臨床病例（一般對話），正常回覆即可，不需套用推理框架"""

TOOLS = [
    {
        "name": "search_evidence",
        "description": "Search OpenEvidence for medical literature. Use for rare diagnoses needing current guidelines, close-probability differentials, or specific treatment protocols.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Clinical question in English (e.g., 'likelihood ratio of S3 gallop for heart failure')",
                }
            },
            "required": ["question"],
        },
    }
]


def _blocks_to_dicts(content) -> list[dict]:
    result = []
    for block in content:
        if block.type == "tool_use":
            result.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
        elif block.type == "text":
            result.append({"type": "text", "text": block.text})
    return result


class OpenEvidenceClient:
    BASE_URL = "https://www.openevidence.com"
    PENDING = {"queued", "pending", "processing", "running", "in_progress"}

    def __init__(self):
        raw = os.environ.get("OE_COOKIES_JSON", "")
        if not raw:
            self._cookie_header = None
            return
        cookies = json.loads(raw)
        pairs = [
            f"{c['name']}={c['value']}"
            for c in cookies
            if "openevidence.com" in c.get("domain", "")
        ]
        self._cookie_header = "; ".join(pairs) if pairs else None

    def _headers(self) -> dict:
        return {
            "cookie": self._cookie_header or "",
            "origin": self.BASE_URL,
            "referer": f"{self.BASE_URL}/",
            "accept": "application/json, text/plain, */*",
            "content-type": "application/json",
        }

    async def ask(self, question: str) -> str:
        if not self._cookie_header:
            return "[文獻搜尋未設定，缺少 OE_COOKIES_JSON]"

        payload = {
            "article_type": "Ask OpenEvidence Light with citations",
            "inputs": {
                "variant_configuration_file": "prod",
                "attachments": [],
                "question": question,
                "use_gatekeeper": True,
            },
            "personalization_enabled": False,
            "disable_caching": False,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.BASE_URL}/api/article", headers=self._headers(), json=payload
            )
            if resp.status_code == 401:
                return "[文獻搜尋暫時無法使用，請更新 Cookies]"
            resp.raise_for_status()
            article_id = resp.json()["id"]

        started = asyncio.get_event_loop().time()
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                if asyncio.get_event_loop().time() - started > 45:
                    return "[文獻搜尋逾時]"
                resp = await client.get(
                    f"{self.BASE_URL}/api/article/{article_id}", headers=self._headers()
                )
                if resp.status_code == 401:
                    return "[文獻搜尋暫時無法使用，請更新 Cookies]"
                article = resp.json()
                status = str(article.get("status", "")).lower()
                if status and status not in self.PENDING:
                    return self._extract_text(article)
                await asyncio.sleep(3)

    def _extract_text(self, article: dict) -> str:
        output = article.get("output") or {}
        structured = output.get("structured_article") or {}
        raw_text = structured.get("raw_text", "")
        if raw_text:
            return raw_text
        return output.get("text") or "[文獻搜尋無結果]"


oe_client = OpenEvidenceClient()


def split_message(text: str, limit: int = 4800) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""

    for para in text.split("\n\n"):
        if not current:
            current = para[:limit]
        elif len(current) + 2 + len(para) <= limit:
            current += "\n\n" + para
        else:
            chunks.append(current)
            if len(chunks) == 4:
                chunks.append(para[:limit])
                return chunks
            current = para[:limit]

    if current:
        chunks.append(current)

    return chunks


async def _claude_create_with_retry(**kwargs):
    for attempt in range(3):
        try:
            return claude.messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            if e.status_code != 529 or attempt == 2:
                raise
            await asyncio.sleep(5 * (attempt + 1))


async def call_claude(user_history: list[dict]) -> tuple[str, list[dict]]:
    """Call Claude with CPS system prompt. Handles one tool_use round if triggered.

    Returns (final_text, updated_history).
    updated_history includes any tool_use/tool_result messages inserted mid-turn.
    """
    response = await _claude_create_with_retry(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=CPS_SYSTEM_PROMPT,
        tools=TOOLS,
        tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        messages=user_history,
    )

    if response.stop_reason != "tool_use":
        text_block = next((b for b in response.content if b.type == "text"), None)
        return (text_block.text if text_block else "[無回覆內容]"), user_history

    tool_block = next(b for b in response.content if b.type == "tool_use")
    evidence = await oe_client.ask(tool_block.input["question"])

    extended = user_history + [
        {"role": "assistant", "content": _blocks_to_dicts(response.content)},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_block.id, "content": evidence}
            ],
        },
    ]

    response2 = await _claude_create_with_retry(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=CPS_SYSTEM_PROMPT,
        tools=TOOLS,
        tool_choice={"type": "none"},
        messages=extended,
    )
    text_block2 = next((b for b in response2.content if b.type == "text"), None)
    return (text_block2.text if text_block2 else "[無回覆內容]"), extended


history: dict[str, list[dict]] = defaultdict(list)
MAX_HISTORY = 20


def verify_signature(body: bytes, signature: str, secret: str = None) -> bool:
    if secret is None:
        secret = LINE_CHANNEL_SECRET
    h = hmac.new(secret.encode(), body, hashlib.sha256).digest()
    expected = base64.b64encode(h).decode()
    return hmac.compare_digest(expected, signature)


def add_message(user_id: str, role: str, content: str | list) -> None:
    history[user_id].append({"role": role, "content": content})
    if len(history[user_id]) > MAX_HISTORY:
        history[user_id] = history[user_id][-MAX_HISTORY:]


async def send_reply(reply_token: str, messages: list[str]) -> None:
    async with httpx.AsyncClient() as http:
        await http.post(
            "https://api.line.me/v2/bot/message/reply",
            headers={
                "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "replyToken": reply_token,
                "messages": [{"type": "text", "text": m} for m in messages],
            },
            timeout=10.0,
        )


async def push_message(user_id: str, messages: list[str]) -> None:
    async with httpx.AsyncClient() as http:
        await http.post(
            "https://api.line.me/v2/bot/message/push",
            headers={
                "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "to": user_id,
                "messages": [{"type": "text", "text": m} for m in messages],
            },
            timeout=10.0,
        )


async def process_event(user_id: str, reply_token: str, user_text: str) -> None:
    if user_text.strip() in ("新病人", "/reset", "reset"):
        history[user_id] = []
        await send_reply(reply_token, ["已清除對話記錄，請開始描述新病人。"])
        return

    add_message(user_id, "user", user_text)

    try:
        text, _ = await call_claude(list(history[user_id]))
    except anthropic.BadRequestError:
        history[user_id] = []
        await push_message(user_id, ["對話記錄出現問題，已重置，請重新傳送你的問題。"])
        return
    except anthropic.APIStatusError:
        await push_message(user_id, ["伺服器暫時過載，請稍後再試。"])
        return

    add_message(user_id, "assistant", text)
    await push_message(user_id, split_message(text))


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

        asyncio.create_task(process_event(user_id, reply_token, user_text))

    return JSONResponse(content={"status": "ok"})
