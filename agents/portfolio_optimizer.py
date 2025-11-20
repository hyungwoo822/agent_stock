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