import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.ticker_service import (
    canonicalize_ticker,
    display_ticker,
    fetch_realtime_price,
    KNOWN_CRYPTO_SYMBOLS
)

def test_canonicalize_crypto_tickers():
    assert canonicalize_ticker("BTC") == "BTC-USD"
    assert canonicalize_ticker("ada") == "ADA-USD"
    assert canonicalize_ticker("Xrp") == "XRP-USD"
    assert canonicalize_ticker("sol") == "SOL-USD"
    assert canonicalize_ticker("ETH") == "ETH-USD"
    assert canonicalize_ticker("DOGE") == "DOGE-USD"
    assert canonicalize_ticker("crypto:btc") == "BTC-USD"

def test_canonicalize_preserves_stocks_and_pairs():
    assert canonicalize_ticker("NVDA") == "NVDA"
    assert canonicalize_ticker("AAPL") == "AAPL"
    assert canonicalize_ticker("MSFT") == "MSFT"
    assert canonicalize_ticker("BTC-USD") == "BTC-USD"
    assert canonicalize_ticker("ETH-USD") == "ETH-USD"

def test_canonicalize_empty():
    assert canonicalize_ticker("") == ""
    assert canonicalize_ticker("   ") == ""

def test_display_ticker():
    assert display_ticker("BTC-USD") == "BTC"
    assert display_ticker("ADA-USD") == "ADA"
    assert display_ticker("XRP-USD") == "XRP"
    assert display_ticker("NVDA") == "NVDA"
    assert display_ticker("AAPL") == "AAPL"
    assert display_ticker("") == ""

def test_fetch_realtime_price_live():
    # Test Bitcoin price fetching
    btc_price, btc_chg, btc_canonical = fetch_realtime_price("BTC")
    assert btc_canonical == "BTC-USD"
    assert btc_price > 1000.0, f"Expected real BTC price (> $1,000), got {btc_price}"

    # Test Cardano price fetching
    ada_price, ada_chg, ada_canonical = fetch_realtime_price("ADA")
    assert ada_canonical == "ADA-USD"
    assert ada_price > 0.01, f"Expected real ADA price (> $0.01), got {ada_price}"

    # Test XRP price fetching
    xrp_price, xrp_chg, xrp_canonical = fetch_realtime_price("XRP")
    assert xrp_canonical == "XRP-USD"
    # Test NVDA stock price fetching
    nvda_price, nvda_chg, nvda_canonical = fetch_realtime_price("NVDA")
    assert nvda_canonical == "NVDA"
    assert nvda_price > 10.0, f"Expected real NVDA price (> $10), got {nvda_price}"

    print(f"BTC price fetched: ${btc_price:,.2f} ({btc_chg:+.2f}%) [{btc_canonical}]")
    print(f"ADA price fetched: ${ada_price:,.4f} ({ada_chg:+.2f}%) [{ada_canonical}]")
    print(f"XRP price fetched: ${xrp_price:,.4f} ({xrp_chg:+.2f}%) [{xrp_canonical}]")
    print(f"NVDA price fetched: ${nvda_price:,.2f} ({nvda_chg:+.2f}%) [{nvda_canonical}]")

if __name__ == "__main__":
    test_canonicalize_crypto_tickers()
    test_canonicalize_preserves_stocks_and_pairs()
    test_canonicalize_empty()
    test_display_ticker()
    test_fetch_realtime_price_live()
    print("\n[SUCCESS] All ticker service unit tests PASSED successfully!")
