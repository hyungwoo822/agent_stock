# human_loop/interface.py
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
from queue import Queue
import json
import uuid
import logging

logger = logging.getLogger(__name__)

class HumanInTheLoop:
    """Human-in-the-loop 인터페이스"""
    
    def __init__(self):
        self.feedback_queue = Queue()
        self.decision_history = []
        self.learning_data = []
        self.approval_patterns = {}
        
    async def request_review(self, decision_context: Dict) -> Dict:
        """사용자 리뷰 요청"""
        review_request = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "context": decision_context,
            "proposed_actions": decision_context.get("decisions", []),
            "risk_analysis": decision_context.get("risk_assessment", {}),
            "market_conditions": decision_context.get("market_sentiment", "neutral"),
            "estimated_impact": self._calculate_impact(decision_context)
        }
        
        # 사용자에게 리뷰 요청 전송
        response = await self._send_to_user(review_request)
        
        # 피드백 저장 및 학습
        self._store_feedback(response)
        self._learn_from_feedback(response)
        
        return response
    
    async def _send_to_user(self, review_request: Dict) -> Dict:
        """사용자에게 리뷰 요청 전송"""
        # 실제 구현에서는 웹소켓, 이메일, 앱 알림 등 사용
        
        print("\n" + "="*50)
        print("🔔 HUMAN REVIEW REQUESTED")
        print("="*50)
        
        print(f"\n📊 Market Sentiment: {review_request['market_conditions']}")
        print(f"⚠️  Risk Level: {review_request['risk_analysis'].get('risk_level', 'N/A')}")
        print(f"💰 Estimated Impact: ${review_request['estimated_impact']:.2f}")
        
        print("\n📋 Proposed Actions:")
        for i, action in enumerate(review_request['proposed_actions'], 1):
            print(f"  {i}. {action['action']} {action.get('quantity', 'N/A')} shares of {action['ticker']} at ${action.get('limit_price', 'N/A')}")
        
        print("\n" + "-"*50)
        
        # 시뮬레이션을 위한 자동 승인 (실제로는 사용자 입력 대기)
        await asyncio.sleep(2)  # 사용자 검토 시간 시뮬레이션
        
        # 사용자 응답 시뮬레이션
        response = {
            "request_id": review_request["id"],
            "approved": True,  # 또는 사용자 입력에 따라
            "modified_decisions": review_request["proposed_actions"],
            "user_comments": "Looks good, proceed with caution on tech stocks",
            "response_time": datetime.now().isoformat(),
            "confidence_level": 0.8
        }
        
        print(f"\n✅ User Response: {'APPROVED' if response['approved'] else 'MODIFIED'}")
        if response.get('user_comments'):
            print(f"💬 Comments: {response['user_comments']}")
        print("="*50 + "\n")
        
        return response
    
    def _calculate_impact(self, context: Dict) -> float:
        """예상 영향 계산"""
        total_value = 0
        
        for decision in context.get("decisions", []):
            value = decision.get("quantity", 0) * decision.get("limit_price", 0)
            total_value += value
        
        return total_value
    
    def _store_feedback(self, feedback: Dict):
        """피드백 저장"""
        self.decision_history.append({
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback,
            "outcome": None  # 나중에 결과 업데이트
        })
        
        # 피드백을 큐에 추가
        self.feedback_queue.put(feedback)
    
    def _learn_from_feedback(self, feedback: Dict):
        """피드백으로부터 학습"""
        # 승인 패턴 학습
        if feedback.get("approved"):
            # 승인된 결정의 특징 저장
            for decision in feedback.get("modified_decisions", []):
                key = f"{decision['action']}_{decision['ticker']}"
                
                if key not in self.approval_patterns:
                    self.approval_patterns[key] = {
                        "approved_count": 0,
                        "rejected_count": 0,
                        "avg_confidence": 0
                    }
                
                self.approval_patterns[key]["approved_count"] += 1
                self.approval_patterns[key]["avg_confidence"] = (
                    self.approval_patterns[key]["avg_confidence"] * 0.9 +
                    feedback.get("confidence_level", 0.5) * 0.1
                )
        else:
            # 거부된 결정 패턴 학습
            for decision in feedback.get("modified_decisions", []):
                key = f"{decision['action']}_{decision['ticker']}"
                
                if key not in self.approval_patterns:
                    self.approval_patterns[key] = {
                        "approved_count": 0,
                        "rejected_count": 0,
                        "avg_confidence": 0
                    }
                
                self.approval_patterns[key]["rejected_count"] += 1
        
        # 학습 데이터 저장
        self.learning_data.append({
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback,
            "patterns": dict(self.approval_patterns)
        })
    
    def get_approval_likelihood(self, decision: Dict) -> float:
        """결정의 승인 가능성 예측"""
        key = f"{decision['action']}_{decision['ticker']}"
        
        if key in self.approval_patterns:
            pattern = self.approval_patterns[key]
            total = pattern["approved_count"] + pattern["rejected_count"]
            
            if total > 0:
                return pattern["approved_count"] / total
        
        return 0.5  # 기본값
    
    def update_outcome(self, request_id: str, outcome: Dict):
        """결과 업데이트"""
        for record in self.decision_history:
            if record.get("feedback", {}).get("request_id") == request_id:
                record["outcome"] = outcome
                
                # 결과를 바탕으로 추가 학습
                if outcome.get("profit_loss", 0) > 0:
                    # 수익이 난 결정 강화
                    self._reinforce_positive_pattern(record["feedback"])
                else:
                    # 손실이 난 결정 약화
                    self._reinforce_negative_pattern(record["feedback"])
                
                break
    
    def _reinforce_positive_pattern(self, feedback: Dict):
        """긍정적 패턴 강화"""
        for decision in feedback.get("modified_decisions", []):
            key = f"{decision['action']}_{decision['ticker']}"
            
            if key in self.approval_patterns:
                self.approval_patterns[key]["avg_confidence"] = min(
                    1.0,
                    self.approval_patterns[key]["avg_confidence"] * 1.1
                )
    
    def _reinforce_negative_pattern(self, feedback: Dict):
        """부정적 패턴 약화"""
        for decision in feedback.get("modified_decisions", []):
            key = f"{decision['action']}_{decision['ticker']}"
            
            if key in self.approval_patterns:
                self.approval_patterns[key]["avg_confidence"] = max(
                    0.0,
                    self.approval_patterns[key]["avg_confidence"] * 0.9
                )
    
    def get_learning_summary(self) -> Dict:
        """학습 요약 반환"""
        total_decisions = len(self.decision_history)
        approved_decisions = sum(
            1 for d in self.decision_history 
            if d.get("feedback", {}).get("approved")
        )
        
        profitable_outcomes = sum(
            1 for d in self.decision_history
            if d.get("outcome", {}).get("profit_loss", 0) > 0
        )
        
        return {
            "total_decisions_reviewed": total_decisions,
            "approval_rate": approved_decisions / total_decisions if total_decisions > 0 else 0,
            "profitable_rate": profitable_outcomes / total_decisions if total_decisions > 0 else 0,
            "top_approved_patterns": sorted(
                self.approval_patterns.items(),
                key=lambda x: x[1]["approved_count"],
                reverse=True
            )[:5],
            "learning_metrics": {
                "patterns_learned": len(self.approval_patterns),
                "avg_confidence": np.mean([
                    p["avg_confidence"] 
                    for p in self.approval_patterns.values()
                ]) if self.approval_patterns else 0
            }
        }