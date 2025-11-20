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

