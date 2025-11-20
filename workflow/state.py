# workflow/state.py
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class WorkflowState:
    """워크플로우 상태 관리"""
    session_id: str
    user_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Market Analysis
    market_analysis: Dict = field(default_factory=dict)
    market_sentiment: str = "neutral"
    
    # Technical Analysis
    technical_analysis: Dict = field(default_factory=dict)
    technical_signals: List[Dict] = field(default_factory=list)
    
    # Fundamental Analysis
    fundamental_analysis: Dict = field(default_factory=dict)
    valuation_metrics: Dict = field(default_factory=dict)
    
    # Portfolio
    current_portfolio: Dict = field(default_factory=dict)
    optimal_allocation: Dict = field(default_factory=dict)
    rebalancing_actions: List[Dict] = field(default_factory=list)
    
    # Risk Management
    risk_assessment: Dict = field(default_factory=dict)
    risk_level: float = 0.5
    position_change: float = 0.0
    
    # Decisions
    pending_decisions: List[Dict] = field(default_factory=list)
    final_decisions: List[Dict] = field(default_factory=list)
    
    # Human in the Loop
    requires_human_review: bool = False
    human_feedback: Optional[Dict] = None
    
    # Execution
    execution_results: Dict = field(default_factory=dict)
    
    # User Preferences
    user_preferences: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """상태를 딕셔너리로 변환"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "market_analysis": self.market_analysis,
            "market_sentiment": self.market_sentiment,
            "technical_analysis": self.technical_analysis,
            "fundamental_analysis": self.fundamental_analysis,
            "current_portfolio": self.current_portfolio,
            "optimal_allocation": self.optimal_allocation,
            "risk_assessment": self.risk_assessment,
            "risk_level": self.risk_level,
            "pending_decisions": self.pending_decisions,
            "requires_human_review": self.requires_human_review
        }

