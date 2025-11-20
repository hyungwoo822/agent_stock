import FinanceDataReader as fdr
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from utils.circuit_breaker import CircuitBreaker
from utils.rate_limiter import RateLimiter
from config.settings import config
import logging

logger = logging.getLogger(__name__)
rate_limiter = RateLimiter()

class YahooFinanceInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    data_type: str = Field(description="Type of data: price, financials, info, news")
    period: str = Field(default="1mo", description="Data period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")

class YahooFinanceTool(BaseTool):
    name = "yahoo_finance"
    description = "Fetch real-time stock data from FinanceDataReader (replacing Yahoo Finance)"
    args_schema = YahooFinanceInput
    
    @CircuitBreaker(failure_threshold=3, recovery_timeout=60)
    @rate_limiter.limit("yahoo", config.API_RATE_LIMIT["yahoo"], 3600)
    def _run(self, ticker: str, data_type: str, period: str = "1mo") -> Dict:
        """FinanceDataReader 데이터 가져오기"""
        try:
            if data_type == "price":
                # 가격 데이터
                start_date = self._get_start_date(period)
                df = fdr.DataReader(ticker, start=start_date)
                
                if df.empty:
                    return {"error": "No price data found", "ticker": ticker}
                
                current_price = df['Close'].iloc[-1]
                
                # Calculate change
                if len(df) > 1:
                    change_percent = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100)
                else:
                    change_percent = 0
                
                # Convert index to string for JSON serialization
                df.index = df.index.strftime('%Y-%m-%d')
                
                return {
                    "ticker": ticker,
                    "current_price": float(current_price),
                    "history": df.to_dict(),
                    "volume": int(df['Volume'].iloc[-1]) if 'Volume' in df.columns and len(df) > 0 else 0,
                    "change_percent": change_percent
                }
                
            elif data_type == "financials":
                # FDR doesn't support financials directly in the same way
                logger.warning("FinanceDataReader does not support detailed financials. Returning empty.")
                return {
                    "income_statement": {},
                    "balance_sheet": {},
                    "cash_flow": {},
                    "key_metrics": {}
                }
                
            elif data_type == "info":
                # FDR doesn't support company info directly
                logger.warning("FinanceDataReader does not support company info. Returning minimal data.")
                return {
                    "company_name": ticker,
                    "sector": "Unknown",
                    "industry": "Unknown",
                    "market_cap": 0,
                    "description": f"Data for {ticker}"
                }
                
            elif data_type == "news":
                # FDR doesn't support news
                logger.warning("FinanceDataReader does not support news. Returning empty.")
                return {
                    "ticker": ticker,
                    "news": []
                }
                
        except Exception as e:
            logger.error(f"FinanceDataReader error for {ticker}: {e}")
            return {"error": str(e), "ticker": ticker}

    def _get_start_date(self, period: str) -> str:
        """기간 문자열을 시작 날짜로 변환"""
        now = datetime.now()
        if period == "1d":
            start = now - timedelta(days=1)
        elif period == "5d":
            start = now - timedelta(days=5)
        elif period == "1mo":
            start = now - timedelta(days=30)
        elif period == "3mo":
            start = now - timedelta(days=90)
        elif period == "6mo":
            start = now - timedelta(days=180)
        elif period == "1y":
            start = now - timedelta(days=365)
        elif period == "2y":
            start = now - timedelta(days=730)
        elif period == "5y":
            start = now - timedelta(days=1825)
        elif period == "10y":
            start = now - timedelta(days=3650)
        elif period == "ytd":
            start = datetime(now.year, 1, 1)
        else:
            start = now - timedelta(days=30) # Default 1mo
            
        return start.strftime('%Y-%m-%d')

class NewsAggregatorInput(BaseModel):
    keywords: List[str] = Field(description="Keywords to search for")
    sources: List[str] = Field(default=["bbc", "cnn", "reuters", "bloomberg"], description="News sources")
    limit: int = Field(default=10, description="Number of articles to fetch")

class NewsAggregatorTool(BaseTool):
    name = "news_aggregator"
    description = "Aggregate news from multiple sources for market sentiment analysis"
    args_schema = NewsAggregatorInput
    
    @CircuitBreaker(failure_threshold=3, recovery_timeout=60)
    @rate_limiter.limit("news", config.API_RATE_LIMIT["news"], 86400)
    def _run(self, keywords: List[str], sources: List[str] = None, limit: int = 10) -> Dict:
        """뉴스 수집 및 집계"""
        if sources is None:
            sources = ["bbc", "cnn", "reuters", "bloomberg"]
        
        all_articles = []
        
        # NewsAPI 사용
        if config.NEWS_API_KEY:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": " OR ".join(keywords),
                "apiKey": config.NEWS_API_KEY,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": limit
            }
            
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    
                    for article in articles:
                        all_articles.append({
                            "title": article.get('title', ''),
                            "description": article.get('description', ''),
                            "source": article.get('source', {}).get('name', ''),
                            "url": article.get('url', ''),
                            "published_at": article.get('publishedAt', ''),
                            "sentiment": self._analyze_sentiment(article.get('title', '') + " " + article.get('description', ''))
                        })
            except Exception as e:
                logger.error(f"NewsAPI error: {e}")
        
        # 직접 크롤링 (백업)
        if len(all_articles) < limit:
            for source in sources:
                try:
                    articles = self._crawl_news_source(source, keywords, limit - len(all_articles))
                    all_articles.extend(articles)
                except Exception as e:
                    logger.error(f"Error crawling {source}: {e}")
        
        return {
            "keywords": keywords,
            "articles": all_articles[:limit],
            "total_articles": len(all_articles),
            "overall_sentiment": self._calculate_overall_sentiment(all_articles)
        }
    
    def _crawl_news_source(self, source: str, keywords: List[str], limit: int) -> List[Dict]:
        """개별 뉴스 소스 크롤링"""
        articles = []
        
        source_urls = {
            "bbc": "https://www.bbc.com/news/business",
            "cnn": "https://edition.cnn.com/business",
            "reuters": "https://www.reuters.com/markets/",
            "bloomberg": "https://www.bloomberg.com/markets"
        }
        
        if source not in source_urls:
            return articles
        
        try:
            response = requests.get(source_urls[source], timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 각 소스별 파싱 로직
            if source == "bbc":
                article_tags = soup.find_all('article', limit=limit)
                for article in article_tags:
                    title_tag = article.find('h3')
                    if title_tag and any(keyword.lower() in title_tag.text.lower() for keyword in keywords):
                        articles.append({
                            "title": title_tag.text,
                            "source": "BBC",
                            "url": f"https://www.bbc.com{article.find('a')['href']}" if article.find('a') else "",
                            "published_at": datetime.now().isoformat(),
                            "sentiment": self._analyze_sentiment(title_tag.text)
                        })
            
            elif source == "reuters":
                article_tags = soup.find_all('div', class_='media-story', limit=limit)
                for article in article_tags:
                    title_tag = article.find('h3')
                    if title_tag and any(keyword.lower() in title_tag.text.lower() for keyword in keywords):
                        articles.append({
                            "title": title_tag.text,
                            "source": "Reuters",
                            "url": f"https://www.reuters.com{article.find('a')['href']}" if article.find('a') else "",
                            "published_at": datetime.now().isoformat(),
                            "sentiment": self._analyze_sentiment(title_tag.text)
                        })
            
        except Exception as e:
            logger.error(f"Error crawling {source}: {e}")
        
        return articles
    
    def _analyze_sentiment(self, text: str) -> str:
        """간단한 감성 분석"""
        positive_words = ['gain', 'rise', 'up', 'high', 'positive', 'growth', 'profit', 'surge', 'rally', 'boom']
        negative_words = ['loss', 'fall', 'down', 'low', 'negative', 'decline', 'deficit', 'crash', 'plunge', 'crisis']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_overall_sentiment(self, articles: List[Dict]) -> Dict:
        """전체 감성 계산"""
        sentiments = [article.get('sentiment', 'neutral') for article in articles]
        
        return {
            "positive": sentiments.count('positive'),
            "negative": sentiments.count('negative'),
            "neutral": sentiments.count('neutral'),
            "overall": max(set(sentiments), key=sentiments.count) if sentiments else 'neutral'
        }

