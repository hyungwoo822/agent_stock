from typing import Dict, List, Optional
from datetime import datetime, timedelta
from tools.market_tools import YahooFinanceTool, NewsAggregatorTool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config.settings import config
import asyncio
import logging

logger = logging.getLogger(__name__)

class MarketIntelligenceAgent:
    """시장 정보 수집 및 분석 에이전트"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.3,
            api_key=config.OPENAI_API_KEY
        )
        self.yahoo_tool = YahooFinanceTool()
        self.news_tool = NewsAggregatorTool()
        
    async def analyze_market(self, tickers: List[str], sectors: List[str] = None) -> Dict:
        """종합 시장 분석"""
        # 병렬로 데이터 수집
        tasks = []
        
        for ticker in tickers:
            tasks.append(self._get_ticker_data(ticker))
        
        # 뉴스 수집
        if sectors:
            for sector in sectors:
                tasks.append(self._get_sector_news(sector))
        
        results = await asyncio.gather(*tasks)
        
        # 결과 종합
        market_data = {
            "timestamp": datetime.now().isoformat(),
            "tickers_analysis": {},
            "sector_news": {},
            "market_sentiment": {},
            "key_events": []
        }
        
        for result in results:
            if "ticker" in result:
                market_data["tickers_analysis"][result["ticker"]] = result
            elif "sector" in result:
                market_data["sector_news"][result["sector"]] = result
        
        # 시장 심리 분석
        market_data["market_sentiment"] = self._analyze_market_sentiment(market_data)
        
        # 주요 이벤트 추출
        market_data["key_events"] = self._extract_key_events(market_data)
        
        return market_data
    
    async def _get_ticker_data(self, ticker: str) -> Dict:
        """개별 종목 데이터 수집"""
        try:
            # 가격 데이터
            price_data = self.yahoo_tool._run(ticker, "price", "1mo")
            
            # 뉴스
            news_data = self.yahoo_tool._run(ticker, "news", "1mo")
            
            # 기업 정보
            info_data = self.yahoo_tool._run(ticker, "info", "1mo")
            
            return {
                "ticker": ticker,
                "price": price_data,
                "news": news_data,
                "info": info_data,
                "analysis_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}
    
    async def _get_sector_news(self, sector: str) -> Dict:
        """섹터별 뉴스 수집"""
        try:
            keywords = self._get_sector_keywords(sector)
            news = self.news_tool._run(keywords, limit=20)
            
            return {
                "sector": sector,
                "news": news,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching news for {sector}: {e}")
            return {"sector": sector, "error": str(e)}
    
    def _get_sector_keywords(self, sector: str) -> List[str]:
        """섹터별 키워드 생성"""
        sector_keywords = {
            "technology": ["tech stocks", "semiconductor", "software", "AI", "cloud computing"],
            "healthcare": ["pharmaceutical", "biotech", "medical devices", "healthcare"],
            "finance": ["banks", "financial services", "insurance", "fintech"],
            "energy": ["oil", "gas", "renewable energy", "solar", "wind"],
            "consumer": ["retail", "consumer goods", "e-commerce", "consumer spending"]
        }
        
        return sector_keywords.get(sector.lower(), [sector])
    
    def _analyze_market_sentiment(self, market_data: Dict) -> Dict:
        """시장 심리 분석"""
        prompt = ChatPromptTemplate.from_template("""
        Analyze the overall market sentiment based on the following data:
        
        Market Data: {market_data}
        
        Provide:
        1. Overall sentiment (Bullish/Bearish/Neutral)
        2. Key driving factors
        3. Risk factors to watch
        4. Opportunities identified
        5. Sentiment score (0-100, where 0 is extremely bearish and 100 is extremely bullish)
        
        Format as JSON.
        """)
        
        response = self.llm.invoke(prompt.format(market_data=str(market_data)[:3000]))
        
        # Parse response (실제로는 JSON 파싱 필요)
        return {
            "overall": "neutral",
            "score": 50,
            "factors": ["mixed earnings", "inflation concerns"],
            "risks": ["geopolitical tensions"],
            "opportunities": ["tech sector rebound"]
        }
    
    def _extract_key_events(self, market_data: Dict) -> List[Dict]:
        """주요 이벤트 추출"""
        events = []
        
        # 가격 급변동 체크
        for ticker, data in market_data["tickers_analysis"].items():
            if "price" in data and "change_percent" in data["price"]:
                if abs(data["price"]["change_percent"]) > 5:
                    events.append({
                        "type": "price_movement",
                        "ticker": ticker,
                        "change": data["price"]["change_percent"],
                        "severity": "high" if abs(data["price"]["change_percent"]) > 10 else "medium"
                    })
        
        # 중요 뉴스 체크
        for sector, news_data in market_data["sector_news"].items():
            if "news" in news_data and "articles" in news_data["news"]:
                for article in news_data["news"]["articles"][:3]:
                    if article.get("sentiment") in ["positive", "negative"]:
                        events.append({
                            "type": "news",
                            "sector": sector,
                            "headline": article["title"],
                            "sentiment": article["sentiment"]
                        })
        
        return events
