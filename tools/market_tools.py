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

# tools/analysis_tools.py
import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Tuple
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicatorsInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    indicators: List[str] = Field(description="List of indicators: RSI, MACD, BB, SMA, EMA, STOCH")
    period: int = Field(default=14, description="Period for indicators")

class TechnicalIndicatorsTool(BaseTool):
    name = "technical_indicators"
    description = "Calculate technical indicators for stock analysis"
    args_schema = TechnicalIndicatorsInput
    
    def _run(self, ticker: str, indicators: List[str], period: int = 14) -> Dict:
        """기술적 지표 계산"""
        try:
            # Yahoo Finance에서 데이터 가져오기
            stock = yf.Ticker(ticker)
            df = stock.history(period="3mo")
            
            if df.empty:
                return {"error": "No data available for ticker"}
            
            results = {
                "ticker": ticker,
                "indicators": {},
                "signals": []
            }
            
            # RSI
            if "RSI" in indicators:
                rsi = ta.momentum.RSIIndicator(df['Close'], window=period)
                rsi_value = rsi.rsi().iloc[-1]
                results["indicators"]["RSI"] = {
                    "value": rsi_value,
                    "signal": self._get_rsi_signal(rsi_value)
                }
            
            # MACD
            if "MACD" in indicators:
                macd = ta.trend.MACD(df['Close'])
                macd_line = macd.macd().iloc[-1]
                signal_line = macd.macd_signal().iloc[-1]
                histogram = macd.macd_diff().iloc[-1]
                
                results["indicators"]["MACD"] = {
                    "macd": macd_line,
                    "signal": signal_line,
                    "histogram": histogram,
                    "signal": self._get_macd_signal(macd_line, signal_line, histogram)
                }
            
            # Bollinger Bands
            if "BB" in indicators:
                bb = ta.volatility.BollingerBands(df['Close'], window=20)
                upper = bb.bollinger_hband().iloc[-1]
                middle = bb.bollinger_mavg().iloc[-1]
                lower = bb.bollinger_lband().iloc[-1]
                current_price = df['Close'].iloc[-1]
                
                results["indicators"]["BollingerBands"] = {
                    "upper": upper,
                    "middle": middle,
                    "lower": lower,
                    "current_price": current_price,
                    "signal": self._get_bb_signal(current_price, upper, middle, lower)
                }
            
            # Moving Averages
            if "SMA" in indicators:
                sma_20 = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator().iloc[-1]
                sma_50 = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator().iloc[-1]
                sma_200 = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator().iloc[-1] if len(df) > 200 else None
                
                results["indicators"]["SMA"] = {
                    "SMA_20": sma_20,
                    "SMA_50": sma_50,
                    "SMA_200": sma_200,
                    "signal": self._get_ma_signal(df['Close'].iloc[-1], sma_20, sma_50, sma_200)
                }
            
            if "EMA" in indicators:
                ema_20 = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator().iloc[-1]
                ema_50 = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator().iloc[-1]
                
                results["indicators"]["EMA"] = {
                    "EMA_20": ema_20,
                    "EMA_50": ema_50,
                    "signal": self._get_ema_signal(df['Close'].iloc[-1], ema_20, ema_50)
                }
            
            # Stochastic
            if "STOCH" in indicators:
                stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
                k = stoch.stoch().iloc[-1]
                d = stoch.stoch_signal().iloc[-1]
                
                results["indicators"]["Stochastic"] = {
                    "K": k,
                    "D": d,
                    "signal": self._get_stoch_signal(k, d)
                }
            
            # 종합 신호
            results["overall_signal"] = self._get_overall_signal(results["indicators"])
            
            return results
            
        except Exception as e:
            logger.error(f"Technical analysis error for {ticker}: {e}")
            return {"error": str(e), "ticker": ticker}
    
    def _get_rsi_signal(self, rsi: float) -> str:
        """RSI 신호 해석"""
        if rsi > 70:
            return "Overbought - Consider selling"
        elif rsi < 30:
            return "Oversold - Consider buying"
        else:
            return "Neutral"
    
    def _get_macd_signal(self, macd: float, signal: float, histogram: float) -> str:
        """MACD 신호 해석"""
        if macd > signal and histogram > 0:
            return "Bullish - Buy signal"
        elif macd < signal and histogram < 0:
            return "Bearish - Sell signal"
        else:
            return "Neutral"
    
    def _get_bb_signal(self, price: float, upper: float, middle: float, lower: float) -> str:
        """볼린저 밴드 신호 해석"""
        if price > upper:
            return "Overbought - Price above upper band"
        elif price < lower:
            return "Oversold - Price below lower band"
        else:
            return "Neutral - Price within bands"
    
    def _get_ma_signal(self, price: float, sma_20: float, sma_50: float, sma_200: Optional[float]) -> str:
        """이동평균 신호 해석"""
        if price > sma_20 > sma_50:
            return "Bullish - Uptrend"
        elif price < sma_20 < sma_50:
            return "Bearish - Downtrend"
        else:
            return "Neutral"
    
    def _get_ema_signal(self, price: float, ema_20: float, ema_50: float) -> str:
        """EMA 신호 해석"""
        if ema_20 > ema_50 and price > ema_20:
            return "Strong Buy"
        elif ema_20 < ema_50 and price < ema_20:
            return "Strong Sell"
        else:
            return "Hold"
    
    def _get_stoch_signal(self, k: float, d: float) -> str:
        """스토캐스틱 신호 해석"""
        if k > 80 and d > 80:
            return "Overbought"
        elif k < 20 and d < 20:
            return "Oversold"
        elif k > d:
            return "Bullish crossover"
        elif k < d:
            return "Bearish crossover"
        else:
            return "Neutral"
    
    def _get_overall_signal(self, indicators: Dict) -> str:
        """종합 신호 계산"""
        buy_signals = 0
        sell_signals = 0
        
        for indicator, data in indicators.items():
            signal = data.get('signal', '')
            if 'buy' in signal.lower() or 'bullish' in signal.lower() or 'oversold' in signal.lower():
                buy_signals += 1
            elif 'sell' in signal.lower() or 'bearish' in signal.lower() or 'overbought' in signal.lower():
                sell_signals += 1
        
        if buy_signals > sell_signals * 1.5:
            return "STRONG BUY"
        elif buy_signals > sell_signals:
            return "BUY"
        elif sell_signals > buy_signals * 1.5:
            return "STRONG SELL"
        elif sell_signals > buy_signals:
            return "SELL"
        else:
            return "HOLD"

class ValuationMetricsInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    compare_to_industry: bool = Field(default=True, description="Compare to industry averages")

class ValuationMetricsTool(BaseTool):
    name = "valuation_metrics"
    description = "Calculate and analyze valuation metrics for fundamental analysis"
    args_schema = ValuationMetricsInput
    
    def _run(self, ticker: str, compare_to_industry: bool = True) -> Dict:
        """밸류에이션 지표 분석"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # 기본 밸류에이션 지표
            metrics = {
                "ticker": ticker,
                "current_price": info.get('currentPrice', 0),
                "market_cap": info.get('marketCap', 0),
                "enterprise_value": info.get('enterpriseValue', 0),
                "ratios": {
                    "PE_ratio": info.get('trailingPE', 0),
                    "Forward_PE": info.get('forwardPE', 0),
                    "PEG_ratio": info.get('pegRatio', 0),
                    "Price_to_Book": info.get('priceToBook', 0),
                    "Price_to_Sales": info.get('priceToSalesTrailing12Months', 0),
                    "EV_to_EBITDA": info.get('enterpriseToEbitda', 0),
                    "EV_to_Revenue": info.get('enterpriseToRevenue', 0)
                },
                "profitability": {
                    "ROE": info.get('returnOnEquity', 0),
                    "ROA": info.get('returnOnAssets', 0),
                    "Profit_Margin": info.get('profitMargins', 0),
                    "Operating_Margin": info.get('operatingMargins', 0),
                    "Gross_Margin": info.get('grossMargins', 0)
                },
                "financial_health": {
                    "Current_Ratio": info.get('currentRatio', 0),
                    "Quick_Ratio": info.get('quickRatio', 0),
                    "Debt_to_Equity": info.get('debtToEquity', 0),
                    "Interest_Coverage": info.get('interestCoverage', 0),
                    "Free_Cash_Flow": info.get('freeCashflow', 0)
                },
                "growth": {
                    "Revenue_Growth": info.get('revenueGrowth', 0),
                    "Earnings_Growth": info.get('earningsGrowth', 0),
                    "Quarterly_Earnings_Growth": info.get('earningsQuarterlyGrowth', 0)
                }
            }
            
            # 업종 평균과 비교
            if compare_to_industry:
                industry = info.get('industry', '')
                sector = info.get('sector', '')
                
                metrics["comparison"] = {
                    "sector": sector,
                    "industry": industry,
                    "valuation_assessment": self._assess_valuation(metrics["ratios"], sector)
                }
            
            # 종합 평가
            metrics["overall_assessment"] = self._get_overall_assessment(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Valuation analysis error for {ticker}: {e}")
            return {"error": str(e), "ticker": ticker}
    
    def _assess_valuation(self, ratios: Dict, sector: str) -> str:
        """밸류에이션 평가"""
        # 섹터별 평균 PE (간단한 예시)
        sector_avg_pe = {
            "Technology": 25,
            "Healthcare": 22,
            "Financial Services": 15,
            "Consumer Defensive": 20,
            "Energy": 12,
            "Utilities": 18,
            "Real Estate": 20,
            "Materials": 16,
            "Industrials": 18,
            "Consumer Cyclical": 22,
            "Communication Services": 20
        }
        
        pe = ratios.get('PE_ratio', 0)
        sector_pe = sector_avg_pe.get(sector, 18)
        
        if pe == 0:
            return "Unable to assess - No PE ratio available"
        elif pe < sector_pe * 0.8:
            return "Undervalued compared to sector"
        elif pe > sector_pe * 1.2:
            return "Overvalued compared to sector"
        else:
            return "Fairly valued compared to sector"
    
    def _get_overall_assessment(self, metrics: Dict) -> Dict:
        """종합 평가"""
        score = 0
        max_score = 0
        assessments = []
        
        # PE 평가
        pe = metrics["ratios"].get("PE_ratio", 0)
        if 0 < pe < 15:
            score += 2
            assessments.append("Attractive PE ratio")
        elif 15 <= pe < 25:
            score += 1
            assessments.append("Reasonable PE ratio")
        max_score += 2
        
        # ROE 평가
        roe = metrics["profitability"].get("ROE", 0)
        if roe > 0.20:
            score += 2
            assessments.append("Excellent ROE")
        elif roe > 0.15:
            score += 1
            assessments.append("Good ROE")
        max_score += 2
        
        # Debt/Equity 평가
        de = metrics["financial_health"].get("Debt_to_Equity", 0)
        if 0 <= de < 0.5:
            score += 2
            assessments.append("Low debt levels")
        elif 0.5 <= de < 1:
            score += 1
            assessments.append("Moderate debt levels")
        max_score += 2
        
        # 수익성장 평가
        growth = metrics["growth"].get("Earnings_Growth", 0)
        if growth > 0.20:
            score += 2
            assessments.append("Strong earnings growth")
        elif growth > 0.10:
            score += 1
            assessments.append("Decent earnings growth")
        max_score += 2
        
        percentage_score = (score / max_score * 100) if max_score > 0 else 0
        
        if percentage_score >= 75:
            recommendation = "STRONG BUY"
        elif percentage_score >= 60:
            recommendation = "BUY"
        elif percentage_score >= 40:
            recommendation = "HOLD"
        elif percentage_score >= 25:
            recommendation = "SELL"
        else:
            recommendation = "STRONG SELL"
        
        return {
            "score": f"{score}/{max_score}",
            "percentage": percentage_score,
            "recommendation": recommendation,
            "key_points": assessments
        }

class BacktestingInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    strategy: Dict = Field(description="Trading strategy parameters")
    period: str = Field(default="1y", description="Backtest period")

class BacktestingTool(BaseTool):
    name = "backtesting"
    description = "Backtest trading strategies with historical data"
    args_schema = BacktestingInput
    
    def _run(self, ticker: str, strategy: Dict, period: str = "1y") -> Dict:
        """전략 백테스팅"""
        try:
            # 데이터 가져오기
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                return {"error": "No data available for backtesting"}
            
            # 전략 파라미터
            buy_rsi = strategy.get("buy_rsi", 30)
            sell_rsi = strategy.get("sell_rsi", 70)
            stop_loss = strategy.get("stop_loss", 0.05)
            take_profit = strategy.get("take_profit", 0.15)
            
            # RSI 계산
            rsi = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            
            # 백테스팅 로직
            positions = []
            current_position = None
            capital = 100000  # 초기 자본
            
            for i in range(1, len(df)):
                current_price = df['Close'].iloc[i]
                current_rsi = rsi.iloc[i] if i < len(rsi) else 50
                
                # 매수 신호
                if current_position is None and current_rsi < buy_rsi:
                    current_position = {
                        "entry_date": df.index[i],
                        "entry_price": current_price,
                        "shares": capital / current_price,
                        "stop_loss": current_price * (1 - stop_loss),
                        "take_profit": current_price * (1 + take_profit)
                    }
                
                # 매도 신호
                elif current_position is not None:
                    should_sell = (
                        current_rsi > sell_rsi or
                        current_price <= current_position["stop_loss"] or
                        current_price >= current_position["take_profit"]
                    )
                    
                    if should_sell:
                        exit_reason = "RSI" if current_rsi > sell_rsi else \
                                     "Stop Loss" if current_price <= current_position["stop_loss"] else \
                                     "Take Profit"
                        
                        profit = (current_price - current_position["entry_price"]) * current_position["shares"]
                        profit_percentage = ((current_price - current_position["entry_price"]) / 
                                           current_position["entry_price"] * 100)
                        
                        positions.append({
                            "entry_date": current_position["entry_date"].strftime("%Y-%m-%d"),
                            "exit_date": df.index[i].strftime("%Y-%m-%d"),
                            "entry_price": round(current_position["entry_price"], 2),
                            "exit_price": round(current_price, 2),
                            "profit": round(profit, 2),
                            "profit_percentage": round(profit_percentage, 2),
                            "exit_reason": exit_reason
                        })
                        
                        capital += profit
                        current_position = None
            
            # 결과 계산
            total_trades = len(positions)
            winning_trades = len([p for p in positions if p["profit"] > 0])
            losing_trades = total_trades - winning_trades
            
            total_profit = sum(p["profit"] for p in positions)
            total_return = ((capital - 100000) / 100000 * 100)
            
            # Buy and Hold 비교
            buy_hold_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100)
            
            # 샤프 비율 계산
            if total_trades > 0:
                returns = [p["profit_percentage"] for p in positions]
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            return {
                "ticker": ticker,
                "period": period,
                "strategy_parameters": strategy,
                "performance": {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": round((winning_trades / total_trades * 100) if total_trades > 0 else 0, 2),
                    "total_profit": round(total_profit, 2),
                    "total_return": round(total_return, 2),
                    "final_capital": round(capital, 2),
                    "sharpe_ratio": round(sharpe_ratio, 2),
                    "buy_hold_return": round(buy_hold_return, 2),
                    "outperformance": round(total_return - buy_hold_return, 2)
                },
                "trades": positions[:10],  # 최근 10개 거래만
                "recommendation": self._get_backtest_recommendation(total_return, buy_hold_return, sharpe_ratio)
            }
            
        except Exception as e:
            logger.error(f"Backtesting error for {ticker}: {e}")
            return {"error": str(e), "ticker": ticker}
    
    def _get_backtest_recommendation(self, strategy_return: float, buy_hold_return: float, sharpe_ratio: float) -> str:
        """백테스트 결과 기반 추천"""
        if strategy_return > buy_hold_return * 1.5 and sharpe_ratio > 1:
            return "Excellent strategy - Significantly outperforms buy & hold"
        elif strategy_return > buy_hold_return and sharpe_ratio > 0.5:
            return "Good strategy - Outperforms buy & hold"
        elif strategy_return > 0:
            return "Acceptable strategy - Positive returns but consider improvements"
        else:
            return "Poor strategy - Consider different approach"