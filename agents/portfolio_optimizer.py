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
            # agents/portfolio_optimizer.py (계속)
            sector_allocation[sector] = sector_allocation.get(sector, 0) + position.get("value", 0)
        
        # 집중도 계산
        position_values = [p.get("value", 0) for p in portfolio.values()]
        concentration = max(position_values) / total_value if total_value > 0 else 0
        
        return {
            "total_value": total_value,
            "num_positions": len(portfolio),
            "sector_allocation": sector_allocation,
            "concentration_risk": concentration,
            "top_positions": sorted(
                portfolio.items(),
                key=lambda x: x[1].get("value", 0),
                reverse=True
            )[:5]
        }
    
    def _calculate_efficient_frontier(
        self,
        assets: List[Dict],
        target_return: float,
        risk_tolerance: str
    ) -> Dict:
        """효율적 프론티어 계산 (마코위츠 모델)"""
        if not assets:
            return {}
        
        # 수익률과 공분산 행렬 계산
        returns = []
        for asset in assets:
            expected_return = asset.get("expected_return", 0.1)
            returns.append(expected_return)
        
        returns = np.array(returns)
        n_assets = len(returns)
        
        # 간단한 공분산 행렬 (실제로는 역사적 데이터 필요)
        cov_matrix = np.eye(n_assets) * 0.04  # 대각선: 분산 0.04
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    cov_matrix[i, j] = 0.01  # 상관계수 가정
        
        # 리스크 수준에 따른 제약 설정
        risk_constraints = {
            "conservative": {"min_weight": 0.02, "max_weight": 0.20},
            "moderate": {"min_weight": 0.01, "max_weight": 0.30},
            "aggressive": {"min_weight": 0, "max_weight": 0.40}
        }
        
        constraints_dict = risk_constraints.get(risk_tolerance, risk_constraints["moderate"])
        
        # 최적화 문제 설정
        def portfolio_return(weights):
            return -np.dot(weights, returns)  # 최대화를 위해 음수
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # 제약조건
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 가중치 합 = 1
            {'type': 'ineq', 'fun': lambda x: np.dot(x, returns) - target_return}  # 목표 수익률
        ]
        
        # 경계값
        bounds = tuple(
            (constraints_dict["min_weight"], constraints_dict["max_weight"])
            for _ in range(n_assets)
        )
        
        # 초기값
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # 최적화 실행
        result = minimize(
            portfolio_volatility,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x if result.success else x0
        
        # 결과 포맷팅
        allocation = {}
        for i, asset in enumerate(assets):
            if optimal_weights[i] > 0.01:  # 1% 이상만 포함
                allocation[asset["ticker"]] = {
                    "weight": round(optimal_weights[i], 4),
                    "expected_return": asset.get("expected_return", 0.1),
                    "recommendation": asset.get("recommendation", "HOLD")
                }
        
        return {
            "allocation": allocation,
            "expected_portfolio_return": round(np.dot(optimal_weights, returns), 4),
            "expected_portfolio_volatility": round(portfolio_volatility(optimal_weights), 4),
            "sharpe_ratio": round(
                (np.dot(optimal_weights, returns) - 0.03) / portfolio_volatility(optimal_weights),
                2
            ) if portfolio_volatility(optimal_weights) > 0 else 0
        }
    
    def _generate_rebalancing(
        self,
        current_portfolio: Dict,
        optimal_weights: Dict,
        user_profile: Dict
    ) -> List[Dict]:
        """리밸런싱 제안 생성"""
        if not optimal_weights.get("allocation"):
            return []
        
        total_value = sum(
            position.get("value", 0) 
            for position in current_portfolio.values()
        )
        
        if total_value == 0:
            total_value = 100000  # 기본값 설정
        
        rebalancing_actions = []
        
        # 현재 가중치 계산
        current_weights = {}
        for ticker, position in current_portfolio.items():
            current_weights[ticker] = position.get("value", 0) / total_value
        
        # 최적 가중치와 비교
        for ticker, optimal in optimal_weights["allocation"].items():
            current_weight = current_weights.get(ticker, 0)
            optimal_weight = optimal["weight"]
            
            difference = optimal_weight - current_weight
            
            if abs(difference) > 0.02:  # 2% 이상 차이날 때만
                value_change = difference * total_value
                
                if difference > 0:
                    action = "BUY"
                else:
                    action = "SELL"
                    value_change = abs(value_change)
                
                rebalancing_actions.append({
                    "action": action,
                    "ticker": ticker,
                    "current_weight": round(current_weight * 100, 2),
                    "target_weight": round(optimal_weight * 100, 2),
                    "value": round(value_change, 2),
                    "priority": "HIGH" if abs(difference) > 0.05 else "MEDIUM"
                })
        
        # 우선순위 정렬
        rebalancing_actions.sort(
            key=lambda x: (x["priority"] == "HIGH", abs(x["value"])),
            reverse=True
        )
        
        return rebalancing_actions
    
    def _optimize_for_taxes(
        self,
        current_portfolio: Dict,
        rebalancing_suggestions: List[Dict],
        tax_strategy: str
    ) -> Dict:
        """세금 최적화"""
        tax_optimization = {
            "strategy": tax_strategy,
            "recommendations": [],
            "estimated_tax_impact": 0
        }
        
        if tax_strategy == "tax_loss_harvesting":
            # 손실 실현을 통한 세금 절감
            for ticker, position in current_portfolio.items():
                unrealized_pnl = position.get("unrealized_pnl", 0)
                
                if unrealized_pnl < -1000:  # $1000 이상 손실
                    tax_optimization["recommendations"].append({
                        "action": "HARVEST_LOSS",
                        "ticker": ticker,
                        "loss_amount": abs(unrealized_pnl),
                        "tax_benefit": abs(unrealized_pnl) * 0.25  # 25% 세율 가정
                    })
        
        elif tax_strategy == "long_term_gains":
            # 장기 보유 우선
            for suggestion in rebalancing_suggestions:
                if suggestion["action"] == "SELL":
                    ticker = suggestion["ticker"]
                    holding_period = current_portfolio.get(ticker, {}).get("holding_days", 0)
                    
                    if holding_period < 365:
                        tax_optimization["recommendations"].append({
                            "action": "DELAY_SALE",
                            "ticker": ticker,
                            "days_until_long_term": 365 - holding_period,
                            "potential_tax_savings": suggestion["value"] * 0.10
                        })
        
        # 예상 세금 영향 계산
        for rec in tax_optimization["recommendations"]:
            if "tax_benefit" in rec:
                tax_optimization["estimated_tax_impact"] -= rec["tax_benefit"]
            elif "potential_tax_savings" in rec:
                tax_optimization["estimated_tax_impact"] -= rec["potential_tax_savings"]
        
        return tax_optimization
    
    def _calculate_risk_metrics(self, optimal_weights: Dict, assets: List[Dict]) -> Dict:
        """리스크 지표 계산"""
        if not optimal_weights.get("allocation"):
            return {}
        
        # VaR (Value at Risk) 계산 - 95% 신뢰수준
        volatility = optimal_weights.get("expected_portfolio_volatility", 0)
        var_95 = 1.645 * volatility * 100000  # $100k 포트폴리오 기준
        
        # 최대 낙폭 (간단한 추정)
        max_drawdown = volatility * 2.5  # 대략적 추정
        
        # 베타 계산 (시장 대비)
        portfolio_beta = 1.0  # 간단히 1로 가정
        
        return {
            "volatility": round(volatility * 100, 2),
            "var_95": round(var_95, 2),
            "max_drawdown_estimate": round(max_drawdown * 100, 2),
            "portfolio_beta": portfolio_beta,
            "diversification_ratio": self._calculate_diversification_ratio(optimal_weights)
        }
    
    def _calculate_diversification_ratio(self, optimal_weights: Dict) -> float:
        """분산 비율 계산"""
        if not optimal_weights.get("allocation"):
            return 0
        
        weights = [w["weight"] for w in optimal_weights["allocation"].values()]
        
        # 허핀달-허쉬만 지수 (HHI)
        hhi = sum(w ** 2 for w in weights)
        
        # 분산 비율 (1에 가까울수록 잘 분산됨)
        diversification_ratio = 1 - hhi
        
        return round(diversification_ratio, 3)
    
    def _create_implementation_plan(self, rebalancing_suggestions: List[Dict]) -> Dict:
        """실행 계획 생성"""
        if not rebalancing_suggestions:
            return {"status": "no_action_needed"}
        
        # 거래를 단계별로 그룹화
        phases = {
            "immediate": [],  # 즉시 실행
            "short_term": [],  # 1주 이내
            "medium_term": []  # 1개월 이내
        }
        
        total_trades = len(rebalancing_suggestions)
        
        for i, suggestion in enumerate(rebalancing_suggestions):
            if suggestion["priority"] == "HIGH" or i < total_trades * 0.3:
                phases["immediate"].append(suggestion)
            elif i < total_trades * 0.6:
                phases["short_term"].append(suggestion)
            else:
                phases["medium_term"].append(suggestion)
        
        return {
            "phases": phases,
            "estimated_transaction_costs": len(rebalancing_suggestions) * 10,  # $10 per trade
            "implementation_timeline": "1-4 weeks",
            "monitoring_frequency": "daily",
            "review_schedule": "monthly"
        }
