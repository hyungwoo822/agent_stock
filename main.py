import asyncio
import logging
import sys
from config.user_config import UserConfig
from agents.supervisor import PortfolioSupervisor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading_bot.log')
    ]
)
logger = logging.getLogger(__name__)

async def main():
    try:
        # 1. Get User Configuration
        config = UserConfig.from_input()
        
        # 2. Initialize Supervisor
        supervisor = PortfolioSupervisor(config)
        
        # 3. Start Trading Loop
        await supervisor.start()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())