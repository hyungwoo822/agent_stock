# main.py (완전한 버전)
import os
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import redis
import json
from datetime import datetime

from workflow.portfolio_workflow import PortfolioManagementWorkflow
from memory.session_manager import SessionManager
from config.settings import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Portfolio Management Agent API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session manager
session_manager = SessionManager()

# Redis client for inter-service communication
redis_client = redis.Redis(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    decode_responses=True
)

# Request models
class AnalysisRequest(BaseModel):
    user_id: str
    query: str
    tickers: List[str]
    session_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    session_id: str
    feedback: Dict

class AgentTaskRequest(BaseModel):
    task_id: str
    task_type: str
    parameters: Dict
    priority: int = 5

# API endpoints
@app.post("/api/analyze")
async def analyze_portfolio(request: AnalysisRequest):
    """포트폴리오 분석 실행"""
    try:
        # Create or get session
        session_id = request.session_id
        if not session_id:
            session_id = session_manager.create_session(request.user_id)
        
        # Initialize workflow
        workflow = PortfolioManagementWorkflow(
            user_id=request.user_id,
            session_id=session_id
        )
        
        # Run analysis
        result = await workflow.run(
            user_query=request.query,
            tickers=request.tickers
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """사용자 피드백 제출"""
    try:
        # Get session
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Store feedback
        session_manager.update_session_state(
            request.session_id,
            {"user_feedback": request.feedback}
        )
        
        # Publish feedback to agents
        redis_client.publish(
            "feedback_channel",
            json.dumps({
                "session_id": request.session_id,
                "feedback": request.feedback,
                "timestamp": datetime.now().isoformat()
            })
        )
        
        return {"status": "success", "message": "Feedback received"}
        
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent/task")
async def submit_agent_task(request: AgentTaskRequest):
    """에이전트에 작업 제출"""
    try:
        # Publish task to appropriate agent queue
        task_queue = f"{request.task_type}_queue"
        
        task_data = {
            "task_id": request.task_id,
            "type": request.task_type,
            "parameters": request.parameters,
            "priority": request.priority,
            "timestamp": datetime.now().isoformat()
        }
        
        redis_client.rpush(task_queue, json.dumps(task_data))
        
        logger.info(f"Task {request.task_id} submitted to {task_queue}")
        
        return {
            "status": "submitted",
            "task_id": request.task_id,
            "queue": task_queue
        }
        
    except Exception as e:
        logger.error(f"Task submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agent/task/{task_id}")
async def get_task_status(task_id: str):
    """작업 상태 조회"""
    try:
        # Get task result from Redis
        result_key = f"task_result:{task_id}"
        result = redis_client.get(result_key)
        
        if result:
            return json.loads(result)
        else:
            # Check if task is still pending
            for queue in ["market_intelligence_queue", "technical_analysis_queue", 
                         "fundamental_analysis_queue", "portfolio_optimization_queue"]:
                queue_items = redis_client.lrange(queue, 0, -1)
                for item in queue_items:
                    task = json.loads(item)
                    if task.get("task_id") == task_id:
                        return {"status": "pending", "task_id": task_id}
            
            return {"status": "not_found", "task_id": task_id}
            
    except Exception as e:
        logger.error(f"Task status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """세션 정보 조회"""
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session

@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "service": os.getenv("SERVICE_NAME", "unknown"),
        "redis_connected": redis_client.ping(),
        "timestamp": datetime.now().isoformat()
    }

# Service runners for each agent
class SupervisorService:
    """Supervisor 에이전트 서비스"""
    
    def __init__(self):
        from agents.supervisor import PortfolioSupervisor
        self.supervisor = None
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            decode_responses=True
        )
        self.pubsub = self.redis_client.pubsub()
        
    async def run(self):
        """Supervisor 서비스 실행"""
        logger.info("Starting Supervisor Agent Service...")
        
        # Subscribe to channels
        self.pubsub.subscribe([
            "supervisor_commands",
            "agent_results"
        ])
        
        while True:
            try:
                # Check for messages
                message = self.pubsub.get_message(timeout=1.0)
                
                if message and message['type'] == 'message':
                    await self.handle_message(message)
                
                # Process pending tasks
                await self.process_tasks()
                
                # Health check
                self.redis_client.set("supervisor_heartbeat", datetime.now().isoformat(), ex=30)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Supervisor service error: {e}")
                await asyncio.sleep(5)
    
    async def handle_message(self, message):
        """메시지 처리"""
        try:
            data = json.loads(message['data'])
            channel = message['channel']
            
            if channel == "supervisor_commands":
                await self.handle_command(data)
            elif channel == "agent_results":
                await self.handle_agent_result(data)
                
        except Exception as e:
            logger.error(f"Message handling error: {e}")
    
    async def handle_command(self, command: Dict):
        """명령 처리"""
        cmd_type = command.get("type")
        
        if cmd_type == "plan_strategy":
            # Initialize supervisor for user
            user_id = command.get("user_id")
            session_id = command.get("session_id")
            
            from agents.supervisor import PortfolioSupervisor
            self.supervisor = PortfolioSupervisor(user_id, session_id)
            
            # Plan strategy
            strategy = self.supervisor.plan_strategy(
                command.get("query", ""),
                command.get("market_context", {})
            )
            
            # Publish result
            result = {
                "type": "strategy_result",
                "session_id": session_id,
                "strategy": strategy,
                "timestamp": datetime.now().isoformat()
            }
            
            self.redis_client.publish("strategy_results", json.dumps(result))
            
        elif cmd_type == "make_decision":
            # Make final decision
            decisions = self.supervisor.make_final_decision(
                command.get("agent_results", {}),
                command.get("user_constraints", {})
            )
            
            # Publish decision
            self.redis_client.publish("final_decisions", json.dumps(decisions))
    
    async def handle_agent_result(self, result: Dict):
        """에이전트 결과 처리"""
        logger.info(f"Received agent result: {result.get('agent_name')} - {result.get('task_id')}")
        
        # Store result
        task_id = result.get("task_id")
        if task_id:
            self.redis_client.set(
                f"agent_result:{task_id}",
                json.dumps(result),
                ex=3600
            )
    
    async def process_tasks(self):
        """대기 중인 작업 처리"""
        # Check supervisor queue
        task = self.redis_client.lpop("supervisor_queue")
        if task:
            task_data = json.loads(task)
            logger.info(f"Processing supervisor task: {task_data.get('task_id')}")
            
            # Process based on task type
            if task_data.get("type") == "coordinate":
                # Coordinate agents
                await self.coordinate_agents(task_data)

    async def coordinate_agents(self, task_data: Dict):
        """에이전트 조정"""
        # Distribute tasks to different agents
        tickers = task_data.get("parameters", {}).get("tickers", [])
        
        # Create tasks for each agent
        tasks = [
            {
                "agent": "market_intelligence",
                "task": {
                    "task_id": f"{task_data['task_id']}_market",
                    "type": "analyze_market",
                    "parameters": {"tickers": tickers}
                }
            },
            {
                "agent": "technical_analyst",
                "task": {
                    "task_id": f"{task_data['task_id']}_technical",
                    "type": "technical_analysis",
                    "parameters": {"tickers": tickers}
                }
            },
            {
                "agent": "fundamental_analyst",
                "task": {
                    "task_id": f"{task_data['task_id']}_fundamental",
                    "type": "fundamental_analysis",
                    "parameters": {"tickers": tickers}
                }
            }
        ]
        
        # Submit tasks to agent queues
        for task in tasks:
            queue_name = f"{task['agent']}_queue"
            self.redis_client.rpush(queue_name, json.dumps(task['task']))
            logger.info(f"Task submitted to {queue_name}")

class MarketIntelligenceService:
    """Market Intelligence 에이전트 서비스"""
    
    def __init__(self):
        from agents.market_intelligence import MarketIntelligenceAgent
        self.agent = MarketIntelligenceAgent()
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            decode_responses=True
        )
    
    async def run(self):
        """Market Intelligence 서비스 실행"""
        logger.info("Starting Market Intelligence Agent Service...")
        
        while True:
            try:
                # Process queue
                task = self.redis_client.blpop("market_intelligence_queue", timeout=5)
                
                if task:
                    _, task_data = task
                    task_json = json.loads(task_data)
                    
                    logger.info(f"Processing market task: {task_json.get('task_id')}")
                    
                    # Execute task
                    result = await self.process_task(task_json)
                    
                    # Store result
                    self.redis_client.set(
                        f"task_result:{task_json['task_id']}",
                        json.dumps(result),
                        ex=3600
                    )
                    
                    # Publish completion
                    self.redis_client.publish("agent_results", json.dumps({
                        "agent_name": "market_intelligence",
                        "task_id": task_json['task_id'],
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    }))
                
                # Periodic market update
                await self.periodic_update()
                
                # Health check
                self.redis_client.set("market_intelligence_heartbeat", 
                                    datetime.now().isoformat(), ex=30)
                
            except Exception as e:
                logger.error(f"Market Intelligence service error: {e}")
                await asyncio.sleep(5)
    
    async def process_task(self, task: Dict) -> Dict:
        """작업 처리"""
        task_type = task.get("type")
        parameters = task.get("parameters", {})
        
        if task_type == "analyze_market":
            tickers = parameters.get("tickers", [])
            sectors = parameters.get("sectors", [])
            
            result = await self.agent.analyze_market(tickers, sectors)
            
            return {
                "status": "completed",
                "task_id": task.get("task_id"),
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        
        return {"status": "unknown_task_type", "task_id": task.get("task_id")}
    
    async def periodic_update(self):
        """주기적 시장 업데이트"""
        # Check if update is needed
        last_update = self.redis_client.get("market_last_update")
        
        if last_update:
            last_update_time = datetime.fromisoformat(last_update)
            if (datetime.now() - last_update_time).seconds < config.MARKET_UPDATE_INTERVAL:
                return
        
        # Perform market update
        logger.info("Performing periodic market update...")
        
        # Update major indices
        indices = ["SPY", "QQQ", "DIA", "IWM", "VIX"]
        result = await self.agent.analyze_market(indices, [])
        
        # Store in Redis for other agents
        self.redis_client.set(
            "market_snapshot",
            json.dumps(result),
            ex=config.MARKET_UPDATE_INTERVAL
        )
        
        self.redis_client.set("market_last_update", datetime.now().isoformat())
        
        logger.info("Market update completed")

class TechnicalAnalystService:
    """Technical Analyst 에이전트 서비스"""
    
    def __init__(self):
        from agents.technical_analyst import TechnicalAnalystAgent
        self.agent = TechnicalAnalystAgent()
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            decode_responses=True
        )
    
    async def run(self):
        """Technical Analyst 서비스 실행"""
        logger.info("Starting Technical Analyst Agent Service...")
        
        while True:
            try:
                # Process queue
                task = self.redis_client.blpop("technical_analyst_queue", timeout=5)
                
                if task:
                    _, task_data = task
                    task_json = json.loads(task_data)
                    
                    logger.info(f"Processing technical analysis task: {task_json.get('task_id')}")
                    
                    # Execute task
                    result = await self.process_task(task_json)
                    
                    # Store result
                    self.redis_client.set(
                        f"task_result:{task_json['task_id']}",
                        json.dumps(result),
                        ex=3600
                    )
                    
                    # Publish completion
                    self.redis_client.publish("agent_results", json.dumps({
                        "agent_name": "technical_analyst",
                        "task_id": task_json['task_id'],
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    }))
                
                # Health check
                self.redis_client.set("technical_analyst_heartbeat", 
                                    datetime.now().isoformat(), ex=30)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Technical Analyst service error: {e}")
                await asyncio.sleep(5)
    
    async def process_task(self, task: Dict) -> Dict:
        """작업 처리"""
        task_type = task.get("type")
        parameters = task.get("parameters", {})
        
        if task_type == "technical_analysis":
            tickers = parameters.get("tickers", [])
            timeframe = parameters.get("timeframe", "daily")
            
            results = {}
            for ticker in tickers:
                results[ticker] = self.agent.analyze(ticker, timeframe)
            
            return {
                "status": "completed",
                "task_id": task.get("task_id"),
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
        
        return {"status": "unknown_task_type", "task_id": task.get("task_id")}

class FundamentalAnalystService:
    """Fundamental Analyst 에이전트 서비스"""
    
    def __init__(self):
        from agents.fundamental_analyst import FundamentalAnalystAgent
        self.agent = FundamentalAnalystAgent()
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            decode_responses=True
        )
    
    async def run(self):
        """Fundamental Analyst 서비스 실행"""
        logger.info("Starting Fundamental Analyst Agent Service...")
        
        while True:
            try:
                # Process queue
                task = self.redis_client.blpop("fundamental_analyst_queue", timeout=5)
                
                if task:
                    _, task_data = task
                    task_json = json.loads(task_data)
                    
                    logger.info(f"Processing fundamental analysis task: {task_json.get('task_id')}")
                    
                    # Execute task
                    result = await self.process_task(task_json)
                    
                    # Store result
                    self.redis_client.set(
                        f"task_result:{task_json['task_id']}",
                        json.dumps(result),
                        ex=3600
                    )
                    
                    # Publish completion
                    self.redis_client.publish("agent_results", json.dumps({
                        "agent_name": "fundamental_analyst",
                        "task_id": task_json['task_id'],
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    }))
                
                # Health check
                self.redis_client.set("fundamental_analyst_heartbeat", 
                                    datetime.now().isoformat(), ex=30)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Fundamental Analyst service error: {e}")
                await asyncio.sleep(5)
    
    async def process_task(self, task: Dict) -> Dict:
        """작업 처리"""
        task_type = task.get("type")
        parameters = task.get("parameters", {})
        
        if task_type == "fundamental_analysis":
            tickers = parameters.get("tickers", [])
            include_competitors = parameters.get("include_competitors", True)
            
            results = {}
            for ticker in tickers:
                results[ticker] = self.agent.analyze(ticker, include_competitors)
            
            return {
                "status": "completed",
                "task_id": task.get("task_id"),
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
        
        return {"status": "unknown_task_type", "task_id": task.get("task_id")}

class PortfolioOptimizerService:
    """Portfolio Optimizer 에이전트 서비스"""
    
    def __init__(self):
        self.agents = {}  # user_id별 agent 인스턴스
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            decode_responses=True
        )
    
    async def run(self):
        """Portfolio Optimizer 서비스 실행"""
        logger.info("Starting Portfolio Optimizer Agent Service...")
        
        while True:
            try:
                # Process queue
                task = self.redis_client.blpop("portfolio_optimizer_queue", timeout=5)
                
                if task:
                    _, task_data = task
                    task_json = json.loads(task_data)
                    
                    logger.info(f"Processing portfolio optimization task: {task_json.get('task_id')}")
                    
                    # Execute task
                    result = await self.process_task(task_json)
                    
                    # Store result
                    self.redis_client.set(
                        f"task_result:{task_json['task_id']}",
                        json.dumps(result),
                        ex=3600
                    )
                    
                    # Publish completion
                    self.redis_client.publish("agent_results", json.dumps({
                        "agent_name": "portfolio_optimizer",
                        "task_id": task_json['task_id'],
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    }))
                
                # Health check
                self.redis_client.set("portfolio_optimizer_heartbeat", 
                                    datetime.now().isoformat(), ex=30)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Portfolio Optimizer service error: {e}")
                await asyncio.sleep(5)
    
    async def process_task(self, task: Dict) -> Dict:
        """작업 처리"""
        from agents.portfolio_optimizer import PortfolioOptimizerAgent
        
        task_type = task.get("type")
        parameters = task.get("parameters", {})
        user_id = parameters.get("user_id")
        
        # Get or create agent for user
        if user_id not in self.agents:
            self.agents[user_id] = PortfolioOptimizerAgent(user_id)
        
        agent = self.agents[user_id]
        
        if task_type == "optimize_portfolio":
            current_portfolio = parameters.get("current_portfolio", {})
            candidate_assets = parameters.get("candidate_assets", [])
            market_conditions = parameters.get("market_conditions", {})
            
            result = agent.optimize_portfolio(
                current_portfolio,
                candidate_assets,
                market_conditions
            )
            
            return {
                "status": "completed",
                "task_id": task.get("task_id"),
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        
        return {"status": "unknown_task_type", "task_id": task.get("task_id")}

class ExecutionService:
    """Execution 에이전트 서비스"""
    
    def __init__(self):
        from agents.execution_agent import ExecutionAgent
        self.agent = ExecutionAgent()
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            decode_responses=True
        )
    
    async def run(self):
        """Execution 서비스 실행"""
        logger.info("Starting Execution Agent Service...")
        
        while True:
            try:
                # Process execution queue
                task = self.redis_client.blpop("execution_queue", timeout=5)
                
                if task:
                    _, task_data = task
                    task_json = json.loads(task_data)
                    
                    logger.info(f"Processing execution task: {task_json.get('task_id')}")
                    
                    # Execute trades
                    decisions = task_json.get("parameters", {}).get("decisions", [])
                    mode = task_json.get("parameters", {}).get("mode", "simulation")
                    
                    result = self.agent.execute_trades(decisions, mode)
                    
                    # Store result
                    self.redis_client.set(
                        f"execution_result:{task_json['task_id']}",
                        json.dumps(result),
                        ex=3600
                    )
                    
                    # Publish completion
                    self.redis_client.publish("execution_results", json.dumps({
                        "task_id": task_json['task_id'],
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    }))
                
                # Monitor positions periodically
                await self.monitor_positions()
                
                # Health check
                self.redis_client.set("execution_heartbeat", 
                                    datetime.now().isoformat(), ex=30)
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Execution service error: {e}")
                await asyncio.sleep(5)
    
    async def monitor_positions(self):
        """포지션 모니터링"""
        # Get current prices (simplified)
        current_prices = {}
        
        # Get price data from market snapshot
        market_snapshot = self.redis_client.get("market_snapshot")
        if market_snapshot:
            snapshot_data = json.loads(market_snapshot)
            # Extract prices from snapshot
            for ticker, data in snapshot_data.get("tickers_analysis", {}).items():
                if "price" in data:
                    current_prices[ticker] = data["price"].get("current_price", 0)
        
        # Monitor positions
        if current_prices:
            monitoring_report = self.agent.monitor_positions(current_prices)
            
            # Store monitoring report
            self.redis_client.set(
                "position_monitoring",
                json.dumps(monitoring_report),
                ex=300
            )
            
            # Check for alerts
            if monitoring_report.get("alerts"):
                # Publish alerts
                for alert in monitoring_report["alerts"]:
                    self.redis_client.publish("trading_alerts", json.dumps(alert))

# Service-specific initialization
def run_service():
    """서비스별 실행"""
    service_name = os.getenv("SERVICE_NAME", "api")
    
    if service_name == "api":
        # Run FastAPI
        logger.info("Starting API Service...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    elif service_name == "supervisor":
        # Run supervisor agent
        logger.info("Starting Supervisor Service...")
        service = SupervisorService()
        asyncio.run(service.run())
        
    elif service_name == "market_intelligence":
        # Run market intelligence agent
        logger.info("Starting Market Intelligence Service...")
        service = MarketIntelligenceService()
        asyncio.run(service.run())
        
    elif service_name == "technical_analyst":
        # Run technical analyst agent
        logger.info("Starting Technical Analyst Service...")
        service = TechnicalAnalystService()
        asyncio.run(service.run())
        
    elif service_name == "fundamental_analyst":
        # Run fundamental analyst agent
        logger.info("Starting Fundamental Analyst Service...")
        service = FundamentalAnalystService()
        asyncio.run(service.run())
        
    elif service_name == "portfolio_optimizer":
        # Run portfolio optimizer agent
        logger.info("Starting Portfolio Optimizer Service...")
        service = PortfolioOptimizerService()
        asyncio.run(service.run())
        
    elif service_name == "execution":
        # Run execution agent
        logger.info("Starting Execution Service...")
        service = ExecutionService()
        asyncio.run(service.run())
        
    else:
        logger.error(f"Unknown service: {service_name}")
        raise ValueError(f"Unknown service: {service_name}")

if __name__ == "__main__":
    try:
        run_service()
    except KeyboardInterrupt:
        logger.info("Service shutdown requested")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        raise