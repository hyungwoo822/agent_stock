# main.py
import os
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

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

# Request models
class AnalysisRequest(BaseModel):
    user_id: str
    query: str
    tickers: List[str]
    session_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    session_id: str
    feedback: Dict

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
        
        return {"status": "success", "message": "Feedback received"}
        
    except Exception as e:
        logger.error(f"Feedback error: {e}")
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
    return {"status": "healthy", "service": os.getenv("SERVICE_NAME", "unknown")}

# Service-specific initialization
def run_service():
    """서비스별 실행"""
    service_name = os.getenv("SERVICE_NAME", "api")
    
    if service_name == "api":
        # Run FastAPI
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    elif service_name == "supervisor":
        # Run supervisor agent
        from agents.supervisor import PortfolioSupervisor
        logger.info("Starting Supervisor Agent...")
        # Agent-specific logic
        
    elif service_name == "market_intelligence":
        # Run market intelligence agent
        from agents.market_intelligence import MarketIntelligenceAgent
        logger.info("Starting Market Intelligence Agent...")
        # Agent-specific logic
        
    else:
        logger.error(f"Unknown service: {service_name}")

if __name__ == "__main__":
    run_service()