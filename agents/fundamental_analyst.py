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