import os
import base64
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, UploadFile, File, Form
from pydantic import BaseModel
import anthropic

from finance_db import FinanceDB, CATEGORIES
from stock_prices import get_price

router = APIRouter(prefix="/finance")
db = FinanceDB()

FINANCE_API_KEY = os.environ.get("FINANCE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _auth(x_api_key: Optional[str]):
    if FINANCE_API_KEY and x_api_key != FINANCE_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Transactions ──────────────────────────────────────────────

class TransactionIn(BaseModel):
    date: str
    merchant: str
    amount: int
    currency: str = "TWD"
    category: str
    bank: str
    card_last4: str = ""
    note: str = ""
    is_travel: int = 0


class CategoryUpdate(BaseModel):
    category: str


class NoteUpdate(BaseModel):
    note: str


@router.get("/transactions")
def get_transactions(year_month: Optional[str] = None, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    return db.list_transactions(year_month)


@router.post("/transactions")
def post_transaction(tx: TransactionIn, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    return {"id": db.insert_transaction(tx.model_dump())}


@router.post("/transactions/batch")
def post_transactions_batch(txs: list[TransactionIn], x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    ids = db.insert_transactions_batch([t.model_dump() for t in txs])
    return {"ids": ids, "count": len(ids)}


@router.patch("/transactions/{tx_id}/category")
def patch_category(tx_id: int, body: CategoryUpdate, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    db.update_category(tx_id, body.category)
    return {"ok": True}


@router.patch("/transactions/{tx_id}/note")
def patch_note(tx_id: int, body: NoteUpdate, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    db.update_note(tx_id, body.note)
    return {"ok": True}


@router.delete("/transactions/{tx_id}")
def delete_transaction(tx_id: int, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    db.delete_transaction(tx_id)
    return {"ok": True}


@router.get("/summary/{year_month}")
def get_summary(year_month: str, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    return db.monthly_summary(year_month)


# ── Screenshot import ─────────────────────────────────────────

PARSE_PROMPT = """這是台灣銀行信用卡對帳單截圖（銀行：{bank}）。

請提取所有「消費」交易，輸出 JSON 陣列。每筆格式：
{{"date":"YYYY-MM-DD","merchant":"商家名稱","amount":金額整數,"currency":"TWD","category":"類別","is_travel":0或1}}

類別只能選以下9個之一：
日常（餐廳/超商/便利商店/日用品/醫院掛號費等小額醫療）、房租、交通（Uber/計程車/高鐵/捷運/停車）、
旅遊（國外消費/訂房/旅遊景點/交通/旅遊保險）、娛樂（串流/KTV/電影）、教育（書籍/訂閱學習/學費）、
醫療（醫院大額費用/手術/住院）、贈與（禮品/包裹/婚喪）、長期規劃（保費/定存/大額保險）

規則：
- 負數金額（折抵/退款/回饋金）→ 跳過
- 金額為0 → 跳過
- 「國外交易服務費」「國外交易手續費」→ 跳過
- 點數/里程紀錄 → 跳過
- 英文商家名稱或有原幣（JPY/AUD/USD等）→ is_travel=1，category="旅遊"
- 高鐵/台鐵/捷運/優步/計程車 → category="交通"，is_travel=0
- ICOCA/SUICA（日本IC卡）→ 旅遊
- Booking.com → 旅遊
- Apple.com/BILL/Netflix/Spotify → 娛樂
- Claude.ai → 教育

只輸出 JSON 陣列，不要任何說明文字。"""


@router.post("/import/screenshot")
async def import_screenshot(
    image: UploadFile = File(...),
    bank: str = Form(...),
    year: str = Form(...),
    x_api_key: Optional[str] = Header(None),
):
    """Accept a screenshot image, parse transactions with Claude Vision."""
    _auth(x_api_key)

    img_bytes = await image.read()
    img_b64 = base64.standard_b64encode(img_bytes).decode()
    media_type = image.content_type or "image/jpeg"

    response = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": PARSE_PROMPT.format(bank=bank) + f"\n\n年份：{year}（若截圖只有月/日，補足此年份）",
                    },
                ],
            }
        ],
    )

    import json, re
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        txs = json.loads(raw)
    except json.JSONDecodeError as e:
        logging.error("JSON parse error: %s | raw: %s", e, raw[:300])
        raise HTTPException(status_code=422, detail=f"解析失敗：{e}")

    # Normalize
    for tx in txs:
        tx.setdefault("bank", bank)
        tx.setdefault("card_last4", "")
        tx.setdefault("note", "")
        tx.setdefault("currency", "TWD")
        tx.setdefault("is_travel", 0)
        if tx.get("category") not in CATEGORIES:
            tx["category"] = "日常"

    return {"transactions": txs, "count": len(txs)}


# ── Income ────────────────────────────────────────────────────

class IncomeIn(BaseModel):
    year_month: str
    source: str
    amount: int
    note: str = ""


@router.get("/income")
def get_income(year_month: Optional[str] = None, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    return db.list_income(year_month)


@router.post("/income")
def post_income(inc: IncomeIn, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    db.upsert_income(inc.year_month, inc.source, inc.amount, inc.note)
    return {"ok": True}


@router.delete("/income/{income_id}")
def delete_income(income_id: int, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    db.delete_income(income_id)
    return {"ok": True}


# ── Holdings ──────────────────────────────────────────────────

class HoldingIn(BaseModel):
    market: str
    ticker: str
    name: str
    shares: float


@router.get("/holdings")
def get_holdings(x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    return db.list_holdings()


@router.post("/holdings")
def post_holding(h: HoldingIn, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    db.upsert_holding(h.market, h.ticker, h.name, h.shares)
    return {"ok": True}


@router.delete("/holdings/{holding_id}")
def delete_holding(holding_id: int, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    db.delete_holding(holding_id)
    return {"ok": True}


# ── Liabilities ───────────────────────────────────────────────

class LiabilityIn(BaseModel):
    type: str
    name: str
    balance: int
    monthly_payment: int = 0


@router.get("/liabilities")
def get_liabilities(x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    return db.list_liabilities()


@router.post("/liabilities")
def post_liability(l: LiabilityIn, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    db.upsert_liability(l.type, l.name, l.balance, l.monthly_payment)
    return {"ok": True}


@router.delete("/liabilities/{liability_id}")
def delete_liability(liability_id: int, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    db.delete_liability(liability_id)
    return {"ok": True}


# ── Cash ──────────────────────────────────────────────────────

class CashIn(BaseModel):
    name: str
    balance: int


@router.get("/cash")
def get_cash(x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    return db.list_cash()


@router.post("/cash")
def post_cash(c: CashIn, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    db.upsert_cash(c.name, c.balance)
    return {"ok": True}


@router.delete("/cash/{cash_id}")
def delete_cash_entry(cash_id: int, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    db.delete_cash(cash_id)
    return {"ok": True}


# ── Net Worth ─────────────────────────────────────────────────

class NetWorthSnapshotIn(BaseModel):
    year_month: str
    total_assets: int
    total_liabilities: int


@router.get("/networth/history")
def get_networth_history(x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    return db.list_net_worth_history()


@router.post("/networth/snapshot")
def post_networth_snapshot(s: NetWorthSnapshotIn, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    db.save_net_worth_snapshot(s.year_month, s.total_assets, s.total_liabilities)
    return {"ok": True}


# ── Stock Prices ──────────────────────────────────────────────

@router.get("/stock-price")
def get_stock_price(ticker: str, market: str, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    price = get_price(ticker, market)
    return {"ticker": ticker, "market": market, "price": price}


# ── Categories ────────────────────────────────────────────────

@router.get("/categories")
def get_categories():
    return CATEGORIES
