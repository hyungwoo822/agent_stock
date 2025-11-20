from typing import Dict, List, Optional
from datetime import datetime
from config.settings import config
import logging

logger = logging.getLogger(__name__)

class ExecutionAgent:
    """거래 실행 및 모니터링 에이전트"""
    
    def __init__(self, config):
        self.config = config
        # Initialize execution tools
        self.pending_orders = {}
        self.executed_orders = []
        self.positions = {}
        
    def execute_trades(self, decisions: List[Dict], mode: str = "simulation") -> Dict:
        """거래 실행"""
        execution_results = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "orders": [],
            "failed_orders": [],
            "total_executed": 0,
            "total_failed": 0
        }
        
        for decision in decisions:
            try:
                if mode == "simulation":
                    result = self._simulate_execution(decision)
                else:
                    result = self._live_execution(decision)
                
                if result["status"] == "executed":
                    execution_results["orders"].append(result)
                    execution_results["total_executed"] += 1
                    self.executed_orders.append(result)
                    
                    # 포지션 업데이트
                    self._update_positions(result)
                else:
                    execution_results["failed_orders"].append(result)
                    execution_results["total_failed"] += 1
                    
            except Exception as e:
                logger.error(f"Execution error for {decision}: {e}")
                execution_results["failed_orders"].append({
                    "decision": decision,
                    "error": str(e)
                })
                execution_results["total_failed"] += 1
        
        return execution_results
    
    def _simulate_execution(self, decision: Dict) -> Dict:
        """시뮬레이션 실행"""
        import random
        
        # 시뮬레이션 성공 확률 (High for testing)
        success_rate = 0.98
        
        if random.random() < success_rate:
            # Scalping simulation: minimal slippage
            # Assuming liquid market for major tickers
            slippage_pct = random.uniform(-0.0005, 0.0005)  # ±0.05%
            
            limit_price = decision.get("limit_price")
            # If no limit price (market order), use current price from decision context if available, 
            # or just assume it executed at some price. 
            # Since we don't pass current price here easily, let's assume limit_price is close to market.
            # If limit_price is None, we need a fallback. 
            # Ideally decision should have 'current_price' or 'limit_price'.
            
            base_price = limit_price if limit_price else 100.0 # Fallback if missing
            
            executed_price = base_price * (1 + slippage_pct)
            
            return {
                "order_id": f"SIM_{datetime.now().timestamp()}",
                "status": "executed",
                "ticker": decision["ticker"],
                "action": decision["action"],
                "quantity": decision["quantity"],
                "requested_price": base_price,
                "executed_price": round(executed_price, 2),
                "slippage": round(slippage_pct * 100, 4),
                "commission": 0.0, # Zero commission for sim
                "timestamp": datetime.now().isoformat(),
                "mode": "simulation"
            }
        else:
            return {
                "order_id": f"SIM_{datetime.now().timestamp()}",
                "status": "failed",
                "ticker": decision["ticker"],
                "action": decision["action"],
                "reason": "Simulation rejection (Random)",
                "timestamp": datetime.now().isoformat()
            }
    
    def _live_execution(self, decision: Dict) -> Dict:
        """실제 거래 실행 (브로커 API 연동)"""
        # 실제 구현시 브로커 API 호출
        # 예: Interactive Brokers, Alpaca, TD Ameritrade API
        
        logger.warning("Live execution not implemented - using simulation")
        return self._simulate_execution(decision)
    
    def _update_positions(self, order: Dict):
        """포지션 업데이트"""
        ticker = order["ticker"]
        
        if ticker not in self.positions:
            self.positions[ticker] = {
                "quantity": 0,
                "average_price": 0,
                "total_cost": 0
            }
        
        if order["action"] == "buy":
            # 매수
            old_quantity = self.positions[ticker]["quantity"]
            old_total = self.positions[ticker]["total_cost"]
            
            new_quantity = old_quantity + order["quantity"]
            new_total = old_total + (order["executed_price"] * order["quantity"])
            
            self.positions[ticker]["quantity"] = new_quantity
            self.positions[ticker]["total_cost"] = new_total
            self.positions[ticker]["average_price"] = new_total / new_quantity if new_quantity > 0 else 0
            
        elif order["action"] == "sell":
            # 매도
            self.positions[ticker]["quantity"] -= order["quantity"]
            
            if self.positions[ticker]["quantity"] <= 0:
                # 포지션 청산
                self.positions[ticker]["quantity"] = 0
                self.positions[ticker]["average_price"] = 0
                self.positions[ticker]["total_cost"] = 0
    
    def monitor_positions(self, current_prices: Dict) -> Dict:
        """포지션 모니터링"""
        monitoring_report = {
            "timestamp": datetime.now().isoformat(),
            "positions": [],
            "total_value": 0,
            "total_pnl": 0,
            "alerts": []
        }
        
        for ticker, position in self.positions.items():
            if position["quantity"] > 0:
                current_price = current_prices.get(ticker, position["average_price"])
                position_value = position["quantity"] * current_price
                pnl = position_value - position["total_cost"]
                pnl_percentage = (pnl / position["total_cost"] * 100) if position["total_cost"] > 0 else 0
                
                position_info = {
                    "ticker": ticker,
                    "quantity": position["quantity"],
                    "average_price": round(position["average_price"], 2),
                    "current_price": round(current_price, 2),
                    "position_value": round(position_value, 2),
                    "pnl": round(pnl, 2),
                    "pnl_percentage": round(pnl_percentage, 2)
                }
                
                monitoring_report["positions"].append(position_info)
                monitoring_report["total_value"] += position_value
                monitoring_report["total_pnl"] += pnl
                
                # 알림 체크
                if pnl_percentage < -config.STOP_LOSS_PERCENTAGE * 100:
                    monitoring_report["alerts"].append({
                        "type": "stop_loss_alert",
                        "ticker": ticker,
                        "message": f"{ticker} is down {abs(pnl_percentage):.2f}% - Consider stop loss"
                    })
                elif pnl_percentage > 20:
                    monitoring_report["alerts"].append({
                        "type": "take_profit_alert",
                        "ticker": ticker,
                        "message": f"{ticker} is up {pnl_percentage:.2f}% - Consider taking profits"
                    })
        
        monitoring_report["total_value"] = round(monitoring_report["total_value"], 2)
        monitoring_report["total_pnl"] = round(monitoring_report["total_pnl"], 2)
        
        return monitoring_report
    
    def create_order_queue(self, decisions: List[Dict]) -> List[Dict]:
        """주문 큐 생성 및 우선순위 설정"""
        order_queue = []
        
        for decision in decisions:
            order = {
                "id": f"ORD_{datetime.now().timestamp()}_{decision['ticker']}",
                "ticker": decision["ticker"],
                "action": decision["action"],
                "quantity": decision["quantity"],
                "order_type": "limit",
                "limit_price": decision.get("limit_price"),
                "stop_loss": decision.get("stop_loss"),
                "take_profit": decision.get("take_profit"),
                "priority": self._calculate_priority(decision),
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
            
            order_queue.append(order)
            self.pending_orders[order["id"]] = order
        
        # 우선순위로 정렬
        order_queue.sort(key=lambda x: x["priority"], reverse=True)
        
        return order_queue
    
    def _calculate_priority(self, decision: Dict) -> int:
        """주문 우선순위 계산"""
        priority = 50  # 기본 우선순위
        
        # 액션 타입에 따른 조정
        if decision.get("action") == "sell" and decision.get("reason") == "stop_loss":
            priority = 100  # 손절매 최우선
        elif decision.get("action") == "sell" and decision.get("reason") == "take_profit":
            priority = 80
        elif decision.get("action") == "buy" and decision.get("confidence", 0) > 0.8:
            priority = 70
        
        return priority