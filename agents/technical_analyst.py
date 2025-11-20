from typing import Dict, List, Optional
from tools.analysis_tools import TechnicalIndicatorsTool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config.settings import config
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalystAgent:
    """기술적 분석 전문가 에이전트"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.2,
            api_key=config.OPENAI_API_KEY
        )
        self.indicator_tool = TechnicalIndicatorsTool()
        
    def analyze(self, ticker: str, timeframe: str = "daily") -> Dict:
        """종합 기술적 분석"""
        # 모든 주요 지표 계산
        indicators = self.indicator_tool._run(
            ticker=ticker,
            indicators=["RSI", "MACD", "BB", "SMA", "EMA", "STOCH"],
            period=14
        )
        
        # 패턴 인식
        patterns = self._identify_patterns(ticker, indicators)
        
        # 지지/저항 레벨
        support_resistance = self._calculate_support_resistance(ticker)
        
        # 트렌드 분석
        trend_analysis = self._analyze_trend(indicators)
        
        # 종합 분석
        prompt = ChatPromptTemplate.from_template("""
        As a technical analysis expert, provide a comprehensive analysis based on the following data:
        
        Ticker: {ticker}
        Indicators: {indicators}
        Patterns: {patterns}
        Support/Resistance: {support_resistance}
        Trend: {trend_analysis}
        
        Provide:
        1. Current technical setup assessment
        2. Key levels to watch
        3. Entry and exit points
        4. Risk/Reward ratio
        5. Trading recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
        6. Confidence level (0-100%)
        
        Be specific with price levels and reasoning.
        """)
        
        response = self.llm.invoke(prompt.format(
            ticker=ticker,
            indicators=indicators,
            patterns=patterns,
            support_resistance=support_resistance,
            trend_analysis=trend_analysis
        ))
        
        return {
            "ticker": ticker,
            "indicators": indicators,
            "patterns": patterns,
            "support_resistance": support_resistance,
            "trend": trend_analysis,
            "analysis": response.content,
            "recommendation": self._extract_recommendation(response.content)
        }
    
    def _identify_patterns(self, ticker: str, indicators: Dict) -> List[Dict]:
        """차트 패턴 인식"""
        patterns = []
        
        # 간단한 패턴 인식 로직
        if "indicators" in indicators:
            # Golden Cross / Death Cross
            if "SMA" in indicators["indicators"]:
                sma_data = indicators["indicators"]["SMA"]
                if sma_data.get("SMA_50") and sma_data.get("SMA_200"):
                    if sma_data["SMA_50"] > sma_data["SMA_200"]:
                        patterns.append({
                            "pattern": "Golden Cross",
                            "signal": "Bullish",
                            "strength": "Strong"
                        })
                    elif sma_data["SMA_50"] < sma_data["SMA_200"]:
                        patterns.append({
                            "pattern": "Death Cross",
                            "signal": "Bearish",
                            "strength": "Strong"
                        })
            
            # RSI Divergence
            if "RSI" in indicators["indicators"]:
                rsi_value = indicators["indicators"]["RSI"].get("value", 50)
                if rsi_value < 30:
                    patterns.append({
                        "pattern": "RSI Oversold",
                        "signal": "Potential Bounce",
                        "strength": "Medium"
                    })
                elif rsi_value > 70:
                    patterns.append({
                        "pattern": "RSI Overbought",
                        "signal": "Potential Pullback",
                        "strength": "Medium"
                    })
        
        return patterns
    
    def _calculate_support_resistance(self, ticker: str) -> Dict:
        """지지/저항 레벨 계산"""
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            df = stock.history(period="3mo")
            
            if df.empty:
                return {}
            
            # 피벗 포인트 계산
            high = df['High'].iloc[-1]
            low = df['Low'].iloc[-1]
            close = df['Close'].iloc[-1]
            
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            
            return {
                "pivot": round(pivot, 2),
                "resistance_1": round(r1, 2),
                "resistance_2": round(r2, 2),
                "support_1": round(s1, 2),
                "support_2": round(s2, 2),
                "current_price": round(close, 2)
            }
        except Exception as e:
            logger.error(f"Error calculating support/resistance for {ticker}: {e}")
            return {}
    
    def _analyze_trend(self, indicators: Dict) -> Dict:
        """트렌드 분석"""
        trend = {
            "direction": "neutral",
            "strength": 0,
            "timeframe": "short-term"
        }
        
        if "indicators" in indicators:
            # Moving Average 기반 트렌드
            if "SMA" in indicators["indicators"]:
                signal = indicators["indicators"]["SMA"].get("signal", "")
                if "Uptrend" in signal:
                    trend["direction"] = "bullish"
                    trend["strength"] = 70
                elif "Downtrend" in signal:
                    trend["direction"] = "bearish"
                    trend["strength"] = 70
            
            # MACD 기반 모멘텀
            if "MACD" in indicators["indicators"]:
                signal = indicators["indicators"]["MACD"].get("signal", "")
                if "Bullish" in signal:
                    trend["strength"] = min(100, trend["strength"] + 20)
                elif "Bearish" in signal:
                    trend["strength"] = max(0, trend["strength"] - 20)
        
        return trend
    
    def _extract_recommendation(self, analysis_text: str) -> str:
        """분석 텍스트에서 추천 추출"""
        recommendations = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
        
        for rec in recommendations:
            if rec in analysis_text.upper():
                return rec
        
        return "HOLD"
