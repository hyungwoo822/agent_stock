import FinanceDataReader as fdr
from datetime import datetime, timedelta
import pandas as pd

def test_fdr():
    tickers = ["AAPL", "SPY", "QQQ"]
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"Testing FinanceDataReader with tickers: {tickers}")
    print(f"Start date: {start_date}")
    
    for ticker in tickers:
        try:
            print(f"\nFetching {ticker}...")
            df = fdr.DataReader(ticker, start=start_date)
            
            if df.empty:
                print(f"❌ {ticker}: No data found")
            else:
                current_price = df['Close'].iloc[-1]
                print(f"✅ {ticker}: Success")
                print(f"   Current Price: {current_price}")
                print(f"   Rows: {len(df)}")
                print(f"   Columns: {df.columns.tolist()}")
                
        except Exception as e:
            print(f"❌ {ticker}: Error - {e}")

if __name__ == "__main__":
    test_fdr()
