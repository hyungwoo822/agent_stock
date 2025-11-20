import yfinance as yf
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
    description = "Fetch real-time stock data, financials, and company information from Yahoo Finance"
    args_schema = YahooFinanceInput
    
    @CircuitBreaker(failure_threshold=3, recovery_timeout=60)
    @rate_limiter.limit("yahoo", config.API_RATE_LIMIT["yahoo"], 3600)
    def _run(self, ticker: str, data_type: str, period: str = "1mo") -> Dict:
        """Yahoo Finance 데이터 가져오기"""
        try:
            stock = yf.Ticker(ticker)
            
            if data_type == "price":
                # 가격 데이터
                hist = stock.history(period=period)
                current_price = stock.info.get('currentPrice', 0)
                
                return {
                    "ticker": ticker,
                    "current_price": current_price,
                    "history": hist.to_dict(),
                    "volume": int(hist['Volume'].iloc[-1]) if len(hist) > 0 else 0,
                    "change_percent": ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100) if len(hist) > 1 else 0
                }
                
            elif data_type == "financials":
                # 재무제표
                financials = {
                    "income_statement": stock.income_stmt.to_dict() if hasattr(stock, 'income_stmt') else {},
                    "balance_sheet": stock.balance_sheet.to_dict() if hasattr(stock, 'balance_sheet') else {},
                    "cash_flow": stock.cash_flow.to_dict() if hasattr(stock, 'cash_flow') else {},
                    "key_metrics": {
                        "PE": stock.info.get('trailingPE', 0),
                        "PB": stock.info.get('priceToBook', 0),
                        "EPS": stock.info.get('trailingEps', 0),
                        "ROE": stock.info.get('returnOnEquity', 0),
                        "Debt_to_Equity": stock.info.get('debtToEquity', 0),
                        "Profit_Margin": stock.info.get('profitMargins', 0)
                    }
                }
                return financials
                
            elif data_type == "info":
                # 회사 정보
                info = stock.info
                return {
                    "company_name": info.get('longName', ''),
                    "sector": info.get('sector', ''),
                    "industry": info.get('industry', ''),
                    "market_cap": info.get('marketCap', 0),
                    "employees": info.get('fullTimeEmployees', 0),
                    "description": info.get('longBusinessSummary', ''),
                    "website": info.get('website', ''),
                    "headquarters": f"{info.get('city', '')}, {info.get('country', '')}"
                }
                
            elif data_type == "news":
                # 뉴스
                news = stock.news[:10] if hasattr(stock, 'news') else []
                return {
                    "ticker": ticker,
                    "news": [
                        {
                            "title": item.get('title', ''),
                            "publisher": item.get('publisher', ''),
                            "link": item.get('link', ''),
                            "timestamp": datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat()
                        }
                        for item in news
                    ]
                }
                
        except Exception as e:
            logger.error(f"Yahoo Finance error for {ticker}: {e}")
            return {"error": str(e), "ticker": ticker}

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

