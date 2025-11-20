from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class UserConfig:
    """User configuration for the trading bot"""
    # Trading Parameters
    risk_tolerance: str = "moderate"  # conservative, moderate, aggressive
    target_return: float = 0.15
    max_drawdown: float = 0.20
    
    # System Parameters
    update_interval_minutes: int = 60
    trading_interval_seconds: int = 300  # Minimum time between trades
    
    # Asset Universe
    tickers: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"])
    
    # Mode
    mode: str = "simulation"  # simulation, paper, live
    
    @classmethod
    def from_input(cls):
        """Create config from user input"""
        print("\n=== Trading Bot Configuration ===")
        
        # Risk Tolerance
        risk = input("Risk Tolerance (conservative/moderate/aggressive) [moderate]: ").strip().lower()
        if not risk: risk = "moderate"
        
        # Update Interval
        interval = input("Update Interval (minutes) [60]: ").strip()
        interval = int(interval) if interval.isdigit() else 60
        
        # Trading Interval
        trade_interval = input("Trading Interval (seconds) [300]: ").strip()
        trade_interval = int(trade_interval) if trade_interval.isdigit() else 300
        
        # Tickers
        tickers_str = input("Tickers (comma separated) [SPY,QQQ,AAPL,MSFT,NVDA]: ").strip()
        tickers = [t.strip().upper() for t in tickers_str.split(",")] if tickers_str else ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
        
        return cls(
            risk_tolerance=risk,
            update_interval_minutes=interval,
            trading_interval_seconds=trade_interval,
            tickers=tickers
        )
