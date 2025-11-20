import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from config.user_config import UserConfig
from agents.supervisor import PortfolioSupervisor

@pytest.mark.asyncio
async def test_supervisor_initialization():
    config = UserConfig(
        risk_tolerance="moderate",
        update_interval_minutes=1,
        tickers=["AAPL"]
    )
    supervisor = PortfolioSupervisor(config)
    
    assert supervisor.config.risk_tolerance == "moderate"
    assert supervisor.market_agent is not None
    assert supervisor.technical_agent is not None

@pytest.mark.asyncio
async def test_supervisor_loop_one_cycle():
    config = UserConfig(
        risk_tolerance="moderate",
        update_interval_minutes=0, # Run once then exit (mocked)
        tickers=["AAPL"]
    )
    
    with patch('agents.supervisor.asyncio.sleep', side_effect=KeyboardInterrupt): # Stop after one cycle
        supervisor = PortfolioSupervisor(config)
        
        # Mock agents to avoid real API calls
        supervisor.market_agent.analyze_market = AsyncMock(return_value={"AAPL": {"price": 150}})
        supervisor.technical_agent.analyze = MagicMock(return_value={"signal": "buy"}) # Synchronous
        supervisor.fundamental_agent.analyze = MagicMock(return_value={"signal": "buy"}) # Synchronous
        supervisor.execution_agent.execute_trades = AsyncMock(return_value={"status": "executed"})
        
        # Mock make_decisions to avoid real LLM calls
        supervisor.make_decisions = MagicMock(return_value=[{
            "ticker": "AAPL",
            "action": "buy",
            "quantity": 1,
            "reason": "Test decision"
        }])
        
        try:
            await supervisor.start()
        except KeyboardInterrupt:
            pass
            
        # Verify calls
        supervisor.market_agent.analyze_market.assert_called()
        supervisor.technical_agent.analyze.assert_called_with("AAPL")
        supervisor.execution_agent.execute_trades.assert_called()
