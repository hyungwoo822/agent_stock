from typing import Dict, List, Optional, Any
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from memory.memory_system import HybridMemorySystem
from config.settings import config
import logging

logger = logging.getLogger(__name__)

class PortfolioSupervisor:
    """중앙 조정자 - 전체 투자 프로세스 관리"""
    
    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )
        self.memory_system = HybridMemorySystem(user_id)
        self.state = {
            "current_portfolio": {},
            "pending_decisions": [],
            "risk_assessment": {},
            "market_analysis": {}
        }
        
    def plan_strategy(self, user_query: str, market_context: Dict) -> Dict:
        """투자 전략 계획"""
        # 사용자 프로필 가져오기
        user_profile = self.memory_system.user_profile.get_profile(self.user_id)
        
        # 과거 유사 상황 검색
        relevant_memories = self.memory_system.search_relevant_memory(user_query)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a portfolio management supervisor responsible for creating investment strategies.
            Based on the user's profile, market context, and historical patterns, create a comprehensive investment plan.
            
            Consider:
            1. User's risk tolerance and investment horizon
            2. Current market conditions
            3. Sector rotations and trends
            4. Portfolio diversification needs
            5. Tax implications
            
            Output a structured plan with specific actions and their priorities."""),
            HumanMessage(content=f"""
            User Query: {user_query}
            
            User Profile:
            - Risk Tolerance: {user_profile.get('risk_tolerance')}
            - Investment Horizon: {user_profile.get('investment_horizon')}
            - Target Return: {user_profile.get('target_return')}%
            - Max Drawdown: {user_profile.get('max_drawdown')}%
            - Preferred Sectors: {user_profile.get('preferred_sectors')}
            - Avoided Sectors: {user_profile.get('avoided_sectors')}
            
            Market Context: {market_context}
            
            Relevant Historical Patterns: {relevant_memories[:3]}
            
            Create a strategic plan with:
            1. Market Assessment
            2. Recommended Actions (Buy/Sell/Hold)
            3. Position Sizing
            4. Risk Management Rules
            5. Timeline for Execution
            """)
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        # 전략을 구조화된 형태로 파싱
        strategy = self._parse_strategy(response.content)
        
        # 메모리에 저장
        self.memory_system.add_interaction(
            user_query,
            str(strategy),
            {"importance": 0.8, "type": "strategy_planning"}
        )
        
        return strategy
    
    def _parse_strategy(self, strategy_text: str) -> Dict:
        """전략 텍스트를 구조화된 데이터로 변환"""
        # 간단한 파싱 로직 (실제로는 더 정교하게)
        return {
            "timestamp": datetime.now().isoformat(),
            "market_assessment": self._extract_section(strategy_text, "Market Assessment"),
            "recommended_actions": self._extract_actions(strategy_text),
            "position_sizing": self._extract_section(strategy_text, "Position Sizing"),
            "risk_management": self._extract_section(strategy_text, "Risk Management"),
            "execution_timeline": self._extract_section(strategy_text, "Timeline")
        }
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """텍스트에서 특정 섹션 추출"""
        lines = text.split('\n')
        in_section = False
        section_content = []
        
        for line in lines:
            if section_name in line:
                in_section = True
                continue
            elif in_section and line.strip() and not line.startswith((' ', '\t')):
                break
            elif in_section:
                section_content.append(line)
        
        return '\n'.join(section_content).strip()
    
    def _extract_actions(self, text: str) -> List[Dict]:
        """추천 액션 추출"""
        actions = []
        action_section = self._extract_section(text, "Recommended Actions")
        
        for line in action_section.split('\n'):
            if 'buy' in line.lower():
                actions.append({"action": "buy", "details": line})
            elif 'sell' in line.lower():
                actions.append({"action": "sell", "details": line})
            elif 'hold' in line.lower():
                actions.append({"action": "hold", "details": line})
        
        return actions
    
    def coordinate_agents(self, strategy: Dict) -> Dict:
        """에이전트 조정 및 작업 분배"""
        results = {
            "market_intelligence": None,
            "technical_analysis": None,
            "fundamental_analysis": None,
            "portfolio_optimization": None,
            "risk_assessment": None
        }
        
        # 각 에이전트에게 작업 할당
        # (실제 구현에서는 각 에이전트의 인스턴스를 생성하고 호출)
        
        logger.info(f"Coordinating agents for strategy: {strategy}")
        
        return results
    
    def make_final_decision(self, agent_results: Dict, user_constraints: Dict = None) -> Dict:
        """최종 투자 결정"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are making final investment decisions based on multiple agent analyses.
            Synthesize all inputs to create actionable investment decisions.
            
            Consider:
            1. Technical and fundamental signals alignment
            2. Risk/reward ratios
            3. Portfolio impact
            4. Market timing
            5. User constraints
            
            Output specific trades with exact quantities and limit prices."""),
            HumanMessage(content=f"""
            Agent Results: {agent_results}
            User Constraints: {user_constraints}
            Current Portfolio State: {self.state['current_portfolio']}
            
            Make final decisions for:
            1. Which positions to enter/exit
            2. Exact position sizes
            3. Entry/exit prices
            4. Stop loss and take profit levels
            5. Execution priority
            """)
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        decisions = {
            "timestamp": datetime.now().isoformat(),
            "decisions": self._parse_decisions(response.content),
            "rationale": response.content,
            "risk_level": self._calculate_risk_level(agent_results),
            "expected_return": self._calculate_expected_return(agent_results)
        }
        
        # 결정 사항을 상태에 저장
        self.state['pending_decisions'] = decisions['decisions']
        
        return decisions
    
    def _parse_decisions(self, decision_text: str) -> List[Dict]:
        """결정 텍스트 파싱"""
        # 실제 구현에서는 더 정교한 파싱 필요
        return [
            {
                "action": "buy",
                "ticker": "AAPL",
                "quantity": 100,
                "limit_price": 150.0,
                "stop_loss": 142.5,
                "take_profit": 165.0
            }
        ]
    
    def _calculate_risk_level(self, agent_results: Dict) -> float:
        """리스크 레벨 계산"""
        # 각 에이전트 결과를 종합하여 리스크 계산
        return 0.5  # 예시 값
    
    def _calculate_expected_return(self, agent_results: Dict) -> float:
        """예상 수익률 계산"""
        # 각 에이전트 결과를 종합하여 예상 수익률 계산
        return 0.15  # 예시 값

# agents/market_intelligence.py
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

# agents/technical_analyst.py
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

# agents/fundamental_analyst.py
from typing import Dict, List, Optional
from tools.analysis_tools import ValuationMetricsTool
from tools.market_tools import YahooFinanceTool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config.settings import config
import logging

logger = logging.getLogger(__name__)

class FundamentalAnalystAgent:
    """기본적 분석 전문가 에이전트"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.2,
            api_key=config.OPENAI_API_KEY
        )
        self.valuation_tool = ValuationMetricsTool()
        self.yahoo_tool = YahooFinanceTool()
        
    def analyze(self, ticker: str, include_competitors: bool = True) -> Dict:
        """종합 기본적 분석"""
        # 밸류에이션 지표
        valuation = self.valuation_tool._run(ticker, compare_to_industry=True)
        
        # 재무제표 분석
        financials = self.yahoo_tool._run(ticker, "financials", "1y")
        
        # 회사 정보
        company_info = self.yahoo_tool._run(ticker, "info", "1y")
        
        # 경쟁사 분석
        competitor_analysis = {}
        if include_competitors:
            competitor_analysis = self._analyze_competitors(ticker, company_info)
        
        # DCF 모델링
        dcf_valuation = self._calculate_dcf(financials, valuation)
        
        # 종합 분석
        prompt = ChatPromptTemplate.from_template("""
        As a fundamental analysis expert, provide a comprehensive analysis based on:
        
        Ticker: {ticker}
        Valuation Metrics: {valuation}
        Financial Statements: {financials}
        Company Info: {company_info}
        Competitor Analysis: {competitor_analysis}
        DCF Valuation: {dcf_valuation}
        
        Provide:
        1. Financial Health Assessment (score 0-100)
        2. Growth Prospects (score 0-100)
        3. Competitive Position (strong/moderate/weak)
        4. Fair Value Estimate
        5. Margin of Safety
        6. Investment Thesis
        7. Key Risks
        8. Catalysts for Growth
        9. Final Recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
        
        Be specific and data-driven in your analysis.
        """)
        
        response = self.llm.invoke(prompt.format(
            ticker=ticker,
            valuation=str(valuation)[:2000],
            financials=str(financials)[:2000],
            company_info=str(company_info)[:1000],
            competitor_analysis=str(competitor_analysis)[:1000],
            dcf_valuation=dcf_valuation
        ))
        
        return {
            "ticker": ticker,
            "valuation_metrics": valuation,
            "financials_summary": self._summarize_financials(financials),
            "dcf_valuation": dcf_valuation,
            "competitor_analysis": competitor_analysis,
            "analysis": response.content,
            "recommendation": self._extract_recommendation(response.content)
        }
    
    def _analyze_competitors(self, ticker: str, company_info: Dict) -> Dict:
        """경쟁사 분석"""
        sector = company_info.get("sector", "")
        industry = company_info.get("industry", "")
        
        # 간단한 경쟁사 매핑 (실제로는 더 정교한 로직 필요)
        competitor_map = {
            "Technology": {
                "Apple Inc.": ["MSFT", "GOOGL"],
                "Microsoft Corporation": ["AAPL", "GOOGL"],
                "Alphabet Inc.": ["AAPL", "MSFT", "META"]
            },
            "Consumer Cyclical": {
                "Tesla, Inc.": ["F", "GM", "RIVN"],
                "Amazon.com, Inc.": ["WMT", "TGT", "COST"]
            }
        }
        
        competitors = []
        if sector in competitor_map:
            for company, comp_tickers in competitor_map[sector].items():
                if ticker in comp_tickers or any(t == ticker for t in comp_tickers):
                    competitors = comp_tickers
                    break
        
        competitor_data = {}
        for comp_ticker in competitors[:3]:  # 상위 3개 경쟁사만
            try:
                comp_valuation = self.valuation_tool._run(comp_ticker, compare_to_industry=False)
                competitor_data[comp_ticker] = {
                    "PE": comp_valuation.get("ratios", {}).get("PE_ratio", 0),
                    "MarketCap": comp_valuation.get("market_cap", 0),
                    "ROE": comp_valuation.get("profitability", {}).get("ROE", 0)
                }
            except:
                continue
        
        return {
            "competitors": competitor_data,
            "industry": industry,
            "sector": sector
        }
    
    def _calculate_dcf(self, financials: Dict, valuation: Dict) -> Dict:
        """DCF (Discounted Cash Flow) 계산"""
        try:
            # 간단한 DCF 모델
            free_cash_flow = valuation.get("financial_health", {}).get("Free_Cash_Flow", 0)
            if not free_cash_flow or free_cash_flow <= 0:
                return {"error": "Insufficient cash flow data for DCF"}
            
            # 가정
            growth_rate = valuation.get("growth", {}).get("Revenue_Growth", 0.1)
            terminal_growth = 0.03  # 3% 영구 성장률
            discount_rate = 0.10  # 10% 할인율
            years = 5
            
            # 미래 현금흐름 계산
            projected_fcf = []
            current_fcf = free_cash_flow
            
            for year in range(1, years + 1):
                current_fcf = current_fcf * (1 + growth_rate)
                discounted_fcf = current_fcf / ((1 + discount_rate) ** year)
                projected_fcf.append(discounted_fcf)
            
            # 터미널 밸류
            terminal_fcf = current_fcf * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            discounted_terminal = terminal_value / ((1 + discount_rate) ** years)
            
            # 기업가치
            enterprise_value = sum(projected_fcf) + discounted_terminal
            
            # 주당 가치 (간단한 계산)
            market_cap = valuation.get("market_cap", 0)
            current_price = valuation.get("current_price", 0)
            
            if market_cap and current_price:
                shares_outstanding = market_cap / current_price
                fair_value_per_share = enterprise_value / shares_outstanding
            else:
                fair_value_per_share = 0
            
            return {
                "enterprise_value": round(enterprise_value, 2),
                "fair_value_per_share": round(fair_value_per_share, 2),
                "current_price": current_price,
                "upside_potential": round(((fair_value_per_share - current_price) / current_price * 100), 2) if current_price else 0,
                "assumptions": {
                    "growth_rate": growth_rate,
                    "terminal_growth": terminal_growth,
                    "discount_rate": discount_rate
                }
            }
        except Exception as e:
            logger.error(f"DCF calculation error: {e}")
            return {"error": str(e)}
    
    def _summarize_financials(self, financials: Dict) -> Dict:
        """재무제표 요약"""
        if "key_metrics" in financials:
            return financials["key_metrics"]
        return {}
    
    def _extract_recommendation(self, analysis_text: str) -> str:
        """추천 추출"""
        recommendations = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
        
        for rec in recommendations:
            if rec in analysis_text.upper():
                return rec
        
        return "HOLD"

# agents/portfolio_optimizer.py
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from memory.memory_system import UserProfileStore
from config.settings import config
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizerAgent:
    """포트폴리오 최적화 에이전트"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.2,
            api_key=config.OPENAI_API_KEY
        )
        self.user_profile_store = UserProfileStore()
        
    def optimize_portfolio(
        self,
        current_portfolio: Dict,
        candidate_assets: List[Dict],
        market_conditions: Dict
    ) -> Dict:
        """포트폴리오 최적화"""
        # 사용자 프로필 로드
        user_profile = self.user_profile_store.get_profile(self.user_id)
        
        # 현재 포트폴리오 분석
        current_analysis = self._analyze_current_portfolio(current_portfolio)
        
        # 효율적 프론티어 계산
        optimal_weights = self._calculate_efficient_frontier(
            candidate_assets,
            user_profile.get("target_return", 0.15),
            user_profile.get("risk_tolerance", "moderate")
        )
        
        # 리밸런싱 제안
        rebalancing_suggestions = self._generate_rebalancing(
            current_portfolio,
            optimal_weights,
            user_profile
        )
        
        # 세금 최적화
        tax_optimization = self._optimize_for_taxes(
            current_portfolio,
            rebalancing_suggestions,
            user_profile.get("tax_strategy", "standard")
        )
        
        # AI 기반 추천
        prompt = ChatPromptTemplate.from_template("""
        As a portfolio optimization expert, create an optimal portfolio based on:
        
        User Profile:
        - Risk Tolerance: {risk_tolerance}
        - Target Return: {target_return}%
        - Investment Horizon: {investment_horizon}
        - Preferred Sectors: {preferred_sectors}
        - Avoided Sectors: {avoided_sectors}
        
        Current Portfolio: {current_portfolio}
        Market Conditions: {market_conditions}
        Optimal Weights: {optimal_weights}
        Tax Considerations: {tax_optimization}
        
        Provide:
        1. Recommended portfolio allocation (specific percentages)
        2. Rebalancing actions (buy/sell specific amounts)
        3. Risk assessment of new portfolio
        4. Expected return and volatility
        5. Diversification score
        6. Implementation timeline
        7. Alternative scenarios
        """)
        
        response = self.llm.invoke(prompt.format(
            risk_tolerance=user_profile.get("risk_tolerance"),
            target_return=user_profile.get("target_return") * 100,
            investment_horizon=user_profile.get("investment_horizon"),
            preferred_sectors=user_profile.get("preferred_sectors"),
            avoided_sectors=user_profile.get("avoided_sectors"),
            current_portfolio=current_analysis,
            market_conditions=market_conditions,
            optimal_weights=optimal_weights,
            tax_optimization=tax_optimization
        ))
        
        return {
            "current_analysis": current_analysis,
            "optimal_allocation": optimal_weights,
            "rebalancing_actions": rebalancing_suggestions,
            "tax_optimization": tax_optimization,
            "ai_recommendations": response.content,
            "risk_metrics": self._calculate_risk_metrics(optimal_weights, candidate_assets),
            "implementation_plan": self._create_implementation_plan(rebalancing_suggestions)
        }
    
    def _analyze_current_portfolio(self, portfolio: Dict) -> Dict:
        """현재 포트폴리오 분석"""
        if not portfolio:
            return {"status": "empty", "value": 0, "positions": 0}
        
        total_value = sum(position.get("value", 0) for position in portfolio.values())
        
        # 섹터 분포
        sector_allocation = {}
        for ticker, position in portfolio.items():
            sector = position.get("sector", "Unknown")
            sector_allocation[sector] = sector_allocation.get(sector, 0) + position.get("