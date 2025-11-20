from typing import Dict, Optional, List
from langgraph.graph import Graph, END
from workflow.state import WorkflowState
from agents.supervisor import PortfolioSupervisor
from agents.market_intelligence import MarketIntelligenceAgent
from agents.technical_analyst import TechnicalAnalystAgent
from agents.fundamental_analyst import FundamentalAnalystAgent
from agents.portfolio_optimizer import PortfolioOptimizerAgent
from agents.execution_agent import ExecutionAgent
from human_loop.interface import HumanInTheLoop
from memory.session_manager import SessionManager
import asyncio
from config.settings import config
import logging

logger = logging.getLogger(__name__)

class PortfolioManagementWorkflow:
    """포트폴리오 관리 워크플로우"""
    
    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        
        # Initialize agents
        self.supervisor = PortfolioSupervisor(user_id, session_id)
        self.market_intelligence = MarketIntelligenceAgent()
        self.technical_analyst = TechnicalAnalystAgent()
        self.fundamental_analyst = FundamentalAnalystAgent()
        self.portfolio_optimizer = PortfolioOptimizerAgent(user_id)
        self.execution_agent = ExecutionAgent()
        self.human_loop = HumanInTheLoop()
        self.session_manager = SessionManager()
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> Graph:
        """워크플로우 그래프 구성"""
        workflow = Graph()
        
        # Add nodes
        workflow.add_node("start", self.start_workflow)
        workflow.add_node("market_analysis", self.analyze_market)
        workflow.add_node("technical_analysis", self.perform_technical_analysis)
        workflow.add_node("fundamental_analysis", self.perform_fundamental_analysis)
        workflow.add_node("portfolio_optimization", self.optimize_portfolio)
        workflow.add_node("risk_assessment", self.assess_risk)
        workflow.add_node("decision_making", self.make_decisions)
        workflow.add_node("human_review", self.human_review)
        workflow.add_node("execution", self.execute_trades)
        workflow.add_node("monitoring", self.monitor_results)
        
        # Add edges
        workflow.add_edge("start", "market_analysis")
        
        # Parallel analysis
        workflow.add_edge("market_analysis", "technical_analysis")
        workflow.add_edge("market_analysis", "fundamental_analysis")
        
        # Convergence
        workflow.add_edge("technical_analysis", "portfolio_optimization")
        workflow.add_edge("fundamental_analysis", "portfolio_optimization")
        
        workflow.add_edge("portfolio_optimization", "risk_assessment")
        workflow.add_edge("risk_assessment", "decision_making")
        
        # Conditional edge for human review
        workflow.add_conditional_edges(
            "decision_making",
            self.should_request_human_review,
            {
                True: "human_review",
                False: "execution"
            }
        )
        
        workflow.add_edge("human_review", "execution")
        workflow.add_edge("execution", "monitoring")
        workflow.add_edge("monitoring", END)
        
        # Set entry point
        workflow.set_entry_point("start")
        
        return workflow.compile()
    
    async def run(self, user_query: str, tickers: List[str]) -> Dict:
        """워크플로우 실행"""
        # Initialize state
        state = WorkflowState(
            session_id=self.session_id,
            user_id=self.user_id
        )
        
        # Load user preferences
        user_profile = self.portfolio_optimizer.user_profile_store.get_profile(self.user_id)
        state.user_preferences = user_profile
        
        # Set initial context
        initial_context = {
            "user_query": user_query,
            "tickers": tickers,
            "state": state
        }
        
        # Run workflow
        try:
            result = await self.workflow.arun(initial_context)
            
            # Save session state
            self.session_manager.update_session_state(
                self.session_id,
                state.to_dict()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return {"error": str(e), "state": state.to_dict()}
    
    async def start_workflow(self, context: Dict) -> Dict:
        """워크플로우 시작"""
        state = context["state"]
        user_query = context["user_query"]
        
        # Supervisor planning
        strategy = self.supervisor.plan_strategy(
            user_query,
            {"market_status": "open", "volatility": "moderate"}
        )
        
        state.market_analysis["strategy"] = strategy
        
        logger.info(f"Workflow started for session {self.session_id}")
        
        return context
    
    async def analyze_market(self, context: Dict) -> Dict:
        """시장 분석 노드"""
        state = context["state"]
        tickers = context["tickers"]
        
        # Market intelligence analysis
        market_data = await self.market_intelligence.analyze_market(
            tickers,
            state.user_preferences.get("preferred_sectors", [])
        )
        
        state.market_analysis.update(market_data)
        state.market_sentiment = market_data.get("market_sentiment", {}).get("overall", "neutral")
        
        logger.info(f"Market analysis completed: sentiment={state.market_sentiment}")
        
        return context
    
    async def perform_technical_analysis(self, context: Dict) -> Dict:
        """기술적 분석 노드"""
        state = context["state"]
        tickers = context["tickers"]
        
        technical_results = {}
        for ticker in tickers:
            analysis = self.technical_analyst.analyze(ticker)
            technical_results[ticker] = analysis
            
            # Extract signals
            if analysis.get("recommendation") in ["BUY", "STRONG BUY"]:
                state.technical_signals.append({
                    "ticker": ticker,
                    "signal": "BUY",
                    "strength": analysis.get("confidence", 50)
                })
        
        state.technical_analysis = technical_results
        
        logger.info(f"Technical analysis completed: {len(state.technical_signals)} buy signals")
        
        return context
    
    async def perform_fundamental_analysis(self, context: Dict) -> Dict:
        """기본적 분석 노드"""
        state = context["state"]
        tickers = context["tickers"]
        
        fundamental_results = {}
        for ticker in tickers:
            analysis = self.fundamental_analyst.analyze(ticker)
            fundamental_results[ticker] = analysis
            
            # Store valuation metrics
            if "valuation_metrics" in analysis:
                state.valuation_metrics[ticker] = analysis["valuation_metrics"]
        
        state.fundamental_analysis = fundamental_results
        
        logger.info("Fundamental analysis completed")
        
        return context
    
    async def optimize_portfolio(self, context: Dict) -> Dict:
        """포트폴리오 최적화 노드"""
        state = context["state"]
        
        # Prepare candidate assets
        candidate_assets = []
        for ticker in context["tickers"]:
            tech_rec = state.technical_analysis.get(ticker, {}).get("recommendation", "HOLD")
            fund_rec = state.fundamental_analysis.get(ticker, {}).get("recommendation", "HOLD")
            
            # Simple scoring
            score = 0
            if tech_rec in ["BUY", "STRONG BUY"]:
                score += 1
            if fund_rec in ["BUY", "STRONG BUY"]:
                score += 1
            
            if score > 0:
                candidate_assets.append({
                    "ticker": ticker,
                    "expected_return": 0.15 if score == 2 else 0.10,
                    "recommendation": "BUY" if score == 2 else "HOLD"
                })
        
        # Optimize portfolio
        optimization_result = self.portfolio_optimizer.optimize_portfolio(
            state.current_portfolio,
            candidate_assets,
            state.market_analysis
        )
        
        state.optimal_allocation = optimization_result.get("optimal_allocation", {})
        state.rebalancing_actions = optimization_result.get("rebalancing_actions", [])
        
        logger.info(f"Portfolio optimization completed: {len(state.rebalancing_actions)} actions")
        
        return context
    
    async def assess_risk(self, context: Dict) -> Dict:
        """리스크 평가 노드"""
        state = context["state"]
        
        # Calculate portfolio risk
        risk_metrics = {
            "portfolio_volatility": state.optimal_allocation.get("expected_portfolio_volatility", 0),
            "max_drawdown": 0.20,  # Estimated
            "concentration_risk": len(state.optimal_allocation.get("allocation", {})) < 5,
            "market_risk": state.market_sentiment == "bearish"
        }
        
        # Calculate overall risk level
        risk_factors = 0
        if risk_metrics["portfolio_volatility"] > 0.25:
            risk_factors += 1
        if risk_metrics["concentration_risk"]:
            risk_factors += 1
        if risk_metrics["market_risk"]:
            risk_factors += 1
        
        state.risk_level = min(1.0, risk_factors * 0.33)
        state.risk_assessment = risk_metrics
        
        # Calculate position change
        total_rebalancing = sum(
            abs(action.get("value", 0))
            for action in state.rebalancing_actions
        )
        portfolio_value = sum(
            pos.get("value", 0)
            for pos in state.current_portfolio.values()
        ) or 100000
        
        state.position_change = total_rebalancing / portfolio_value
        
        logger.info(f"Risk assessment completed: risk_level={state.risk_level:.2f}")
        
        return context
    
    async def make_decisions(self, context: Dict) -> Dict:
        """의사결정 노드"""
        state = context["state"]
        
        # Supervisor makes final decisions
        decisions = self.supervisor.make_final_decision(
            {
                "market": state.market_analysis,
                "technical": state.technical_analysis,
                "fundamental": state.fundamental_analysis,
                "risk": state.risk_assessment
            },
            state.user_preferences
        )
        
        state.final_decisions = decisions.get("decisions", [])
        state.pending_decisions = decisions.get("decisions", [])
        
        logger.info(f"Decision making completed: {len(state.final_decisions)} decisions")
        
        return context
    
    def should_request_human_review(self, context: Dict) -> bool:
        """Human review 필요 여부 판단"""
        state = context["state"]
        
        # Check conditions for human review
        requires_review = (
            state.risk_level > config.MAX_RISK_SCORE or
            state.position_change > config.MAX_POSITION_SIZE or
            state.user_preferences.get("always_review", False) or
            any(d.get("value", 0) > 50000 for d in state.final_decisions)  # Large trades
        )
        
        state.requires_human_review = requires_review
        
        logger.info(f"Human review required: {requires_review}")
        
        return requires_review
    
    async def human_review(self, context: Dict) -> Dict:
        """Human review 노드"""
        state = context["state"]
        
        # Request human review
        review_context = {
            "decisions": state.final_decisions,
            "risk_assessment": state.risk_assessment,
            "market_sentiment": state.market_sentiment,
            "rebalancing_actions": state.rebalancing_actions
        }
        
        feedback = await self.human_loop.request_review(review_context)
        
        state.human_feedback = feedback
        
        # Update decisions based on feedback
        if feedback.get("approved"):
            logger.info("Human review approved")
        else:
            # Modify decisions based on feedback
            state.final_decisions = feedback.get("modified_decisions", [])
            logger.info("Human review modified decisions")
        
        return context
    
    async def execute_trades(self, context: Dict) -> Dict:
        """거래 실행 노드"""
        state = context["state"]
        
        # Execute trades
        execution_results = self.execution_agent.execute_trades(
            state.final_decisions,
            mode="simulation"  # or "live"
        )
        
        state.execution_results = execution_results
        
        logger.info(
            f"Trade execution completed: "
            f"{execution_results['total_executed']} executed, "
            f"{execution_results['total_failed']} failed"
        )
        
        return context
    
    async def monitor_results(self, context: Dict) -> Dict:
        """결과 모니터링 노드"""
        state = context["state"]
        
        # Get current prices for monitoring
        current_prices = {}
        for ticker in context["tickers"]:
            # Simplified - in reality, fetch real-time prices
            current_prices[ticker] = 150.0
        
        # Monitor positions
        monitoring_report = self.execution_agent.monitor_positions(current_prices)
        
        # Store in memory for learning
        if state.execution_results.get("orders"):
            episode_data = {
                "decision": state.final_decisions,
                "context": {
                    "market_sentiment": state.market_sentiment,
                    "risk_level": state.risk_level
                },
                "outcome": monitoring_report,
                "profit_loss": monitoring_report.get("total_pnl", 0)
            }
            
            self.supervisor.memory_system.episode_memory.store_episode(
                f"episode_{self.session_id}",
                episode_data
            )
        
        logger.info(f"Monitoring completed: Total P&L = ${monitoring_report.get('total_pnl', 0):.2f}")
        
        return {
            "status": "completed",
            "execution_results": state.execution_results,
            "monitoring_report": monitoring_report,
            "session_id": self.session_id
        }