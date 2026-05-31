import logging
import yfinance as yf


def get_price(ticker: str, market: str) -> float:
    """
    market: 'TW' or 'US'
    TW: '2330' → yfinance ticker '2330.TW', price in TWD
    US: 'AAPL' → yfinance ticker 'AAPL', price in USD
    Returns 0.0 on failure.
    """
    yf_ticker = f"{ticker}.TW" if market == "TW" else ticker
    try:
        data = yf.Ticker(yf_ticker)
        hist = data.history(period="2d")
        if hist.empty:
            logging.warning("yfinance: no data for %s", yf_ticker)
            return 0.0
        return float(hist["Close"].iloc[-1])
    except Exception as e:
        logging.error("yfinance error for %s: %s", yf_ticker, e)
        return 0.0
