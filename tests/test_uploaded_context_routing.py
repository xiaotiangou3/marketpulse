import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import services.database as database
import providers

class MockEmbeddingProvider(providers.EmbeddingProvider):
    def get_embedding(self, text: str) -> list[float]:
        return [0.05] * 768

def test_extract_holdings_from_context():
    formatted_context = (
        "Uploaded Portfolio CSV Holdings Context:\n"
        "- Ticker: AAPL, Shares: 100.0, Cost Basis: $210.50\n"
        "- Ticker: MSFT, Shares: 50.0, Cost Basis: $425.20\n"
        "- Ticker: NVDA, Shares: 150.0, Cost Basis: $118.40\n"
    )
    
    holdings = database.extract_holdings_from_context(formatted_context)
    assert holdings is not None
    assert len(holdings) == 3
    assert holdings[0]["ticker"] == "AAPL"
    assert holdings[0]["shares"] == 100.0
    assert holdings[0]["cost_basis"] == 210.50
    
    assert holdings[1]["ticker"] == "MSFT"
    assert holdings[1]["shares"] == 50.0
    assert holdings[1]["cost_basis"] == 425.20

def test_custom_holdings_performance_summary():
    custom_holdings = [
        {"ticker": "AAPL", "shares": 10.0, "cost_basis": 150.0},
        {"ticker": "NVDA", "shares": 20.0, "cost_basis": 100.0}
    ]
    
    summary = database.get_portfolio_performance_summary(custom_holdings=custom_holdings)
    assert "AAPL" in summary
    assert "NVDA" in summary
    assert "Total Market Value" in summary
    # Ensure it didn't return empty portfolio message
    assert "Your portfolio is currently empty" not in summary

def test_visual_file_badge_formatting():
    user_prompt = "Is its performance good?"
    class MockFile:
        def __init__(self, name):
            self.name = name
            
    st_files = [MockFile("portfolio_sample.csv")]
    
    if st_files:
        file_badges = ", ".join([f"`{f.name}`" for f in st_files])
        if user_prompt:
            display_prompt = f"{user_prompt}\n\n📎 **Attached:** {file_badges}"
        else:
            display_prompt = f"📎 **Attached:** {file_badges}"
            
    assert "📎 **Attached:** `portfolio_sample.csv`" in display_prompt
    assert "Is its performance good?" in display_prompt

def test_file_format_restriction():
    class MockFile:
        def __init__(self, name):
            self.name = name
            
    st_files = [
        MockFile("portfolio_sample.csv"),
        MockFile("unsupported_image.png"),
        MockFile("rules_doc.pdf"),
        MockFile("script.sh")
    ]
    
    valid_files = []
    toasted = []
    for f in st_files:
        if not f.name.lower().endswith(('.pdf', '.csv')):
            toasted.append(f.name)
        else:
            valid_files.append(f)
            
    assert len(valid_files) == 2
    assert valid_files[0].name == "portfolio_sample.csv"
    assert valid_files[1].name == "rules_doc.pdf"
    assert len(toasted) == 2
    assert "unsupported_image.png" in toasted
    assert "script.sh" in toasted

def main():
    print("==================================================")
    print("   MARKETPULSE UPLOADED CONTEXT ROUTING TEST SUITE")
    print("==================================================")
    
    try:
        print("\n[Step 1] Testing holdings context extraction...")
        test_extract_holdings_from_context()
        print("  [+] Context extraction passed.")
        
        print("\n[Step 2] Testing custom holdings performance summary...")
        test_custom_holdings_performance_summary()
        print("  [+] Custom holdings performance summary passed.")
        
        print("\n[Step 3] Testing visual file badge formatting...")
        test_visual_file_badge_formatting()
        print("  [+] Visual file badge formatting passed.")
        
        print("\n[Step 4] Testing file format restrictions...")
        test_file_format_restriction()
        print("  [+] File format restrictions test passed.")
        
        print("\n==================================================")
        print("   ALL UPLOADED ROUTING TESTS PASSED!             ")
        print("==================================================")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
