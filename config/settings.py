import os
from typing import Dict, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    YAHOO_FINANCE_API_KEY: str = os.getenv("YAHOO_FINANCE_API_KEY")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY")
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY")
    
    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    
    # Model Configuration
    LLM_MODEL: str = "gpt-4-turbo-preview"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Update Intervals
    MARKET_UPDATE_INTERVAL: int = 300  # 5 minutes
    NEWS_UPDATE_INTERVAL: int = 3600  # 1 hour
    
    # Risk Parameters
    MAX_POSITION_SIZE: float = 0.2  # 20% of portfolio
    MAX_RISK_SCORE: float = 0.7
    STOP_LOSS_PERCENTAGE: float = 0.05
    
    # Rate Limiting
    API_RATE_LIMIT: Dict[str, int] = field(default_factory=lambda: {
        "yahoo": 100,        # per hour
        "news": 500,         # per day
        "alpha_vantage": 5   # per minute
    })

config = Config()