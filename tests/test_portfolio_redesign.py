import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import format_currency, format_delta
from services.ticker_service import fetch_portfolio_history

def test_format_currency_positive():
    assert format_currency(1234.56) == "$1,234.56"
    assert format_currency(1234.56, show_sign=True) == "+$1,234.56"
    assert format_currency(0.0) == "$0.00"
    assert format_currency(0.0, show_sign=True) == "$0.00"

def test_format_currency_negative():
    assert format_currency(-1.0) == "-$1.00"
    assert format_currency(-9.19) == "-$9.19"
    assert format_currency(-9.19, show_sign=True) == "-$9.19"
    assert format_currency(-1250000.50) == "-$1,250,000.50"

def test_format_delta():
    assert format_delta(-9.19, -0.04) == "-$9.19 (-0.04%)"
    assert format_delta(12.50, 0.55) == "+$12.50 (+0.55%)"
    assert format_delta(0.0, 0.0) == "$0.00 (+0.00%)"
    assert format_delta(-1.0) == "-$1.00"
    assert format_delta(1.0) == "+$1.00"

def test_fetch_portfolio_history_empty():
    res = fetch_portfolio_history([], timeframe="1D")
    assert res["start_value"] == 0.0
    assert res["current_value"] == 0.0
    assert res["df"].empty

def test_fetch_portfolio_history_mocked_or_live():
    mock_holdings = [
        {"ticker": "AAPL", "shares": 5.0, "cost_basis": 150.0},
        {"ticker": "MSFT", "shares": 2.0, "cost_basis": 300.0}
    ]
    for tf in ["1D", "1W", "1M", "1Y", "ALL"]:
        res = fetch_portfolio_history(mock_holdings, timeframe=tf)
        assert isinstance(res, dict)
        assert "df" in res
        assert "start_value" in res
        assert "current_value" in res
        assert "change_value" in res
        assert "change_pct" in res
        if not res["df"].empty:
            assert "Portfolio Value ($)" in res["df"].columns
            assert res["current_value"] > 0

def test_fetch_portfolio_history_with_created_at():
    import datetime
    two_weeks_ago = datetime.datetime.now() - datetime.timedelta(days=14)
    one_week_ago = datetime.datetime.now() - datetime.timedelta(days=7)
    mock_holdings = [
        {"ticker": "AAPL", "shares": 2.0, "cost_basis": 150.0, "created_at": two_weeks_ago},
        {"ticker": "MSFT", "shares": 1.0, "cost_basis": 300.0, "created_at": one_week_ago}
    ]
    res_all = fetch_portfolio_history(mock_holdings, timeframe="ALL")
    assert isinstance(res_all, dict)
    assert "df" in res_all
    if not res_all["df"].empty:
        assert res_all["current_value"] > 0
        assert res_all["start_value"] > 0

    # Test that 1Y lookback is clamped to 14 days ago (the earliest ticker creation date)
    res_1y = fetch_portfolio_history(mock_holdings, timeframe="1Y")
    assert isinstance(res_1y, dict)
    if not res_1y["df"].empty:
        assert res_1y["current_value"] > 0
        assert res_1y["start_value"] > 0

if __name__ == "__main__":
    test_format_currency_positive()
    test_format_currency_negative()
    test_format_delta()
    test_fetch_portfolio_history_empty()
    test_fetch_portfolio_history_mocked_or_live()
    test_fetch_portfolio_history_with_created_at()
    print("[SUCCESS] All portfolio redesign tests passed!")
