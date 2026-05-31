import sqlite3
import os
from typing import Optional

CATEGORIES = ["日常", "房租", "交通", "旅遊", "娛樂", "教育", "醫療", "贈與", "長期規劃"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    merchant TEXT NOT NULL,
    amount INTEGER NOT NULL,
    currency TEXT DEFAULT 'TWD',
    category TEXT NOT NULL,
    bank TEXT NOT NULL,
    card_last4 TEXT DEFAULT '',
    note TEXT DEFAULT '',
    is_travel INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS income (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year_month TEXT NOT NULL,
    source TEXT NOT NULL,
    amount INTEGER NOT NULL,
    note TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS holdings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market TEXT NOT NULL,
    ticker TEXT NOT NULL,
    name TEXT NOT NULL,
    shares REAL NOT NULL,
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(market, ticker)
);
CREATE TABLE IF NOT EXISTS liabilities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    balance INTEGER NOT NULL,
    monthly_payment INTEGER DEFAULT 0,
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(name)
);
CREATE TABLE IF NOT EXISTS cash_accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    balance INTEGER NOT NULL,
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS net_worth_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year_month TEXT NOT NULL UNIQUE,
    total_assets INTEGER NOT NULL,
    total_liabilities INTEGER NOT NULL,
    net_worth INTEGER NOT NULL
);
"""


class FinanceDB:
    def __init__(self, path: Optional[str] = None):
        data_dir = os.environ.get("DATA_DIR", "/data")
        os.makedirs(data_dir, exist_ok=True)
        self.path = path or os.path.join(data_dir, "finance.db")

    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # --- Transactions ---

    def insert_transaction(self, tx: dict) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO transactions (date,merchant,amount,currency,category,bank,card_last4,note,is_travel) "
                "VALUES (:date,:merchant,:amount,:currency,:category,:bank,:card_last4,:note,:is_travel)",
                tx,
            )
            return cur.lastrowid

    def insert_transactions_batch(self, txs: list) -> list:
        return [self.insert_transaction(tx) for tx in txs]

    def list_transactions(self, year_month: Optional[str] = None) -> list:
        with self._conn() as conn:
            if year_month:
                rows = conn.execute(
                    "SELECT * FROM transactions WHERE date LIKE ? ORDER BY date DESC",
                    (f"{year_month}%",),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM transactions ORDER BY date DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def update_category(self, tx_id: int, category: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE transactions SET category=? WHERE id=?", (category, tx_id)
            )

    def update_note(self, tx_id: int, note: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE transactions SET note=? WHERE id=?", (note, tx_id)
            )

    def delete_transaction(self, tx_id: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM transactions WHERE id=?", (tx_id,))

    def monthly_summary(self, year_month: str) -> dict:
        txs = self.list_transactions(year_month)
        by_category: dict = {}
        for tx in txs:
            by_category[tx["category"]] = by_category.get(tx["category"], 0) + tx["amount"]
        total_expense = sum(tx["amount"] for tx in txs)
        income_rows = self.list_income(year_month)
        total_income = sum(r["amount"] for r in income_rows)
        return {
            "year_month": year_month,
            "total_expense": total_expense,
            "total_income": total_income,
            "balance": total_income - total_expense,
            "by_category": by_category,
            "transaction_count": len(txs),
        }

    # --- Income ---

    def upsert_income(self, year_month: str, source: str, amount: int, note: str = ""):
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM income WHERE year_month=? AND source=?",
                (year_month, source),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE income SET amount=?,note=? WHERE id=?",
                    (amount, note, existing["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO income (year_month,source,amount,note) VALUES (?,?,?,?)",
                    (year_month, source, amount, note),
                )

    def delete_income(self, income_id: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM income WHERE id=?", (income_id,))

    def list_income(self, year_month: Optional[str] = None) -> list:
        with self._conn() as conn:
            if year_month:
                rows = conn.execute(
                    "SELECT * FROM income WHERE year_month=?", (year_month,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM income ORDER BY year_month DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    # --- Holdings ---

    def upsert_holding(self, market: str, ticker: str, name: str, shares: float):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO holdings (market,ticker,name,shares) VALUES (?,?,?,?) "
                "ON CONFLICT(market,ticker) DO UPDATE SET name=excluded.name, "
                "shares=excluded.shares, updated_at=datetime('now')",
                (market, ticker, name, shares),
            )

    def delete_holding(self, holding_id: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM holdings WHERE id=?", (holding_id,))

    def list_holdings(self) -> list:
        with self._conn() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM holdings ORDER BY market,ticker"
                ).fetchall()
            ]

    # --- Liabilities ---

    def upsert_liability(self, type_: str, name: str, balance: int, monthly_payment: int = 0):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO liabilities (type,name,balance,monthly_payment) VALUES (?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET type=excluded.type, balance=excluded.balance, "
                "monthly_payment=excluded.monthly_payment, updated_at=datetime('now')",
                (type_, name, balance, monthly_payment),
            )

    def delete_liability(self, liability_id: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM liabilities WHERE id=?", (liability_id,))

    def list_liabilities(self) -> list:
        with self._conn() as conn:
            return [
                dict(r) for r in conn.execute("SELECT * FROM liabilities").fetchall()
            ]

    # --- Cash ---

    def upsert_cash(self, name: str, balance: int):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO cash_accounts (name,balance) VALUES (?,?) "
                "ON CONFLICT(name) DO UPDATE SET balance=excluded.balance, updated_at=datetime('now')",
                (name, balance),
            )

    def delete_cash(self, cash_id: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM cash_accounts WHERE id=?", (cash_id,))

    def list_cash(self) -> list:
        with self._conn() as conn:
            return [
                dict(r) for r in conn.execute("SELECT * FROM cash_accounts").fetchall()
            ]

    # --- Net Worth History ---

    def save_net_worth_snapshot(self, year_month: str, assets: int, liabilities: int):
        net = assets - liabilities
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO net_worth_history (year_month,total_assets,total_liabilities,net_worth) "
                "VALUES (?,?,?,?) ON CONFLICT(year_month) DO UPDATE SET "
                "total_assets=excluded.total_assets, total_liabilities=excluded.total_liabilities, "
                "net_worth=excluded.net_worth",
                (year_month, assets, liabilities, net),
            )

    def list_net_worth_history(self) -> list:
        with self._conn() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM net_worth_history ORDER BY year_month"
                ).fetchall()
            ]
