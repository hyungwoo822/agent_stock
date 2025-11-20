# Portfolio Agent System

A sophisticated multi-agent system for portfolio management that combines market intelligence, technical analysis, fundamental analysis, and portfolio optimization — all with human-in-the-loop decision making.

## 📐 System Architecture
```
portfolio-agent/
├── agents/                    
│   ├── supervisor.py           # Orchestrates all agents
│   ├── market_intelligence.py  # Real-time market data & news
│   ├── technical_analyst.py    # Chart patterns & technical indicators
│   ├── fundamental_analyst.py  # Company financials & valuations
│   ├── portfolio_optimizer.py  # Risk-adjusted portfolio construction
│   └── execution_agent.py      # Trade execution & order management
│
├── memory/
│   ├── memory_system.py        # Vector-based memory
│   └── session_manager.py      # Session lifecycle management
│
├── rag/
│   ├── rag_system.py           # Retrieval-augmented generation
│   └── loaders.py              # Document loaders
│
├── tools/
│   ├── market_tools.py         # Market data fetching
│   ├── analysis_tools.py       # Utilities for analysis
│   └── execution_tools.py      # Trade execution tools
│
├── workflow/
│   ├── portfolio_workflow.py   # Main LangGraph workflow
│   └── state.py                # State schema
│
├── human_loop/
│   └── interface.py            # Human-in-the-loop approval system
│
├── utils/
│   ├── circuit_breaker.py      # Fault tolerance
│   ├── rate_limiter.py         # API rate limiting
│   └── encryption.py           # Secure data handling
│
├── config/
│   └── settings.py             # Environment settings
│
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## ✨ Core Features

### 1. Multi-Agent System
- **Supervisor Agent** orchestrates all specialists
- **Market Intelligence**: real-time market data, news, sentiment
- **Technical Analyst**: indicators, chart patterns
- **Fundamental Analyst**: company valuations, financial statements
- **Portfolio Optimizer**: risk-adjusted construction (MPT)
- **Execution Agent**: trade execution & order management

### 2. Memory & Context
- **Conversation Memory** (recent context)
- **Execution Memory** (past trades)
- **Knowledge Base** (research, documents)
- Vector similarity search for retrieval

### 3. Safety & Reliability
- Circuit breaker stops cascading failures
- API rate limiting
- Encryption for sensitive data
- Human approval required for:
  - Large trades
  - Portfolio rebalancing
  - High-risk decisions
  - Stop-loss changes

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Market data API keys

### 🛠 Installation
```bash
git clone 
cd portfolio-agent

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### ⚙ Configuration

Edit `config/settings.py` or use environment variables:
```python
# API Keys
OPENAI_API_KEY=your_key
MARKET_DATA_API_KEY=your_key

# Memory Backend
VECTOR_STORE_TYPE=s3  # or 'opensearch'
AWS_REGION=us-east-1

# Risk Parameters
MAX_POSITION_SIZE=0.1
MAX_PORTFOLIO_RISK=0.15
```

### ▶ Running the System

**Docker:**
```bash
docker-compose up -d
```

**Local:**
```bash
python -m workflow.portfolio_workflow
```

## 📘 Usage Examples

### Basic Portfolio Analysis
```python
from workflow.portfolio_workflow import PortfolioWorkflow

workflow = PortfolioWorkflow()

result = workflow.run({
    "user_query": "Analyze my tech portfolio and suggest rebalancing",
    "portfolio": current_holdings
})
```

### With Human-in-the-Loop
```python
from human_loop.interface import HumanLoopInterface

interface = HumanLoopInterface()

result = interface.execute_with_approval(
    action="rebalance_portfolio",
    recommendations=agent_recommendations
)
```

## 🔁 Agent Workflow
```
User Query → Supervisor → [Parallel Analysis]
                           ├─ Market Intelligence
                           ├─ Technical Analysis
                           └─ Fundamental Analysis
           
Analysis Results → Portfolio Optimizer → Human Review → Execution
```

## 🧠 Memory System

- **Conversation Memory** → chat & analysis history
- **Execution Memory** → trade history
- **Knowledge Base** → research documents

Vector similarity search pulls relevant prior analyses.

## 🛡 Safety Features

- **Circuit Breaker**: stops trading on repeated failures
- **Rate Limiting**: controls API usage
- **Human-in-the-Loop**: required for dangerous or expensive actions

## 🧪 Development

### Adding New Agents

1. Create agent in `agents/`
2. Add tools in `tools/`
3. Register in `workflow/portfolio_workflow.py`
4. Update state schema in `workflow/state.py`

### Testing
```bash
pytest tests/
pytest --cov=. tests/
```

## 📊 Monitoring & Observability

- LangSmith tracing
- Structured logging
- Error alerts
- Performance metrics

## 🔐 Security

- API keys via env
- Encrypted sensitive data
- Trade audit logs
- Role-based approvals

## ⚡ Performance

- Parallel agent execution
- Caching for common queries
- Optimized vector search
- Smart rate limiting

## 📈 Future Enhancements

- [ ] Backtesting framework
- [ ] Multi-portfolio support
- [ ] Advanced risk models (VaR, CVaR)
- [ ] More brokerage integrations
- [ ] Mobile approval app
- [ ] Advanced sentiment analysis

## 🤝 Contributing

Contributions welcome! Please read `CONTRIBUTING.md` before submitting PRs.

## 📄 License

MIT License — see `LICENSE`.

## 🆘 Support

- **GitHub Issues**: [link]
- **Documentation**: [link]
- **Email**: support@example.com