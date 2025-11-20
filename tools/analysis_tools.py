import pandas as pd
import numpy as np
import ta
import FinanceDataReader as fdr
from typing import Dict, List, Optional, Tuple
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def get_start_date(period: str) -> str:
    """기간 문자열을 시작 날짜로 변환"""
    now = datetime.now()
    if period == "1d":
        start = now - timedelta(days=1)
    elif period == "5d":
        start = now - timedelta(days=5)
    elif period == "1mo":
        start = now - timedelta(days=30)
    elif period == "3mo":
        start = now - timedelta(days=90)
    elif period == "6mo":
        start = now - timedelta(days=180)
    elif period == "1y":
        start = now - timedelta(days=365)
    elif period == "2y":
        start = now - timedelta(days=730)
    elif period == "5y":
        start = now - timedelta(days=1825)
    elif period == "10y":
        start = now - timedelta(days=3650)
    elif period == "ytd":
        start = datetime(now.year, 1, 1)
    else:
        start = now - timedelta(days=90) # Default 3mo for analysis
        
    return start.strftime('%Y-%m-%d')

class TechnicalIndicatorsInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    indicators: List[str] = Field(description="List of indicators: RSI, MACD, BB, SMA, EMA, STOCH")
    period: int = Field(default=14, description="Period for indicators")

class TechnicalIndicatorsTool(BaseTool):
    name = "technical_indicators"
    description = "Calculate technical indicators for stock analysis"
    args_schema = TechnicalIndicatorsInput
    
    def _run(self, ticker: str, indicators: List[str], period: int = 14) -> Dict:
        """기술적 지표 계산"""
        try:
            # FinanceDataReader에서 데이터 가져오기
            start_date = get_start_date("3mo") # Enough data for indicators
            df = fdr.DataReader(ticker, start=start_date)
            
            if df.empty:
                return {"error": "No data available for ticker"}
            
            results = {
                "ticker": ticker,
                "indicators": {},
                "signals": []
            }
            
            # RSI
            if "RSI" in indicators:
                rsi = ta.momentum.RSIIndicator(df['Close'], window=period)
                rsi_value = rsi.rsi().iloc[-1]
                results["indicators"]["RSI"] = {
                    "value": rsi_value,
                    "signal": self._get_rsi_signal(rsi_value)
                }
            
            # MACD
            if "MACD" in indicators:
                macd = ta.trend.MACD(df['Close'])
                macd_line = macd.macd().iloc[-1]
                signal_line = macd.macd_signal().iloc[-1]
                histogram = macd.macd_diff().iloc[-1]
                
                results["indicators"]["MACD"] = {
                    "macd": macd_line,
                    "signal": signal_line,
                    "histogram": histogram,
                    "signal": self._get_macd_signal(macd_line, signal_line, histogram)
                }
            
            # Bollinger Bands
            if "BB" in indicators:
                bb = ta.volatility.BollingerBands(df['Close'], window=20)
                upper = bb.bollinger_hband().iloc[-1]
                middle = bb.bollinger_mavg().iloc[-1]
                lower = bb.bollinger_lband().iloc[-1]
                current_price = df['Close'].iloc[-1]
                
                results["indicators"]["BollingerBands"] = {
                    "upper": upper,
                    "middle": middle,
                    "lower": lower,
                    "current_price": current_price,
                    "signal": self._get_bb_signal(current_price, upper, middle, lower)
                }
            
            # Moving Averages
            if "SMA" in indicators:
                sma_20 = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator().iloc[-1]
                sma_50 = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator().iloc[-1]
                sma_200 = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator().iloc[-1] if len(df) > 200 else None
                
                results["indicators"]["SMA"] = {
                    "SMA_20": sma_20,
                    "SMA_50": sma_50,
                    "SMA_200": sma_200,
                    "signal": self._get_ma_signal(df['Close'].iloc[-1], sma_20, sma_50, sma_200)
                }
            
            if "EMA" in indicators:
                ema_20 = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator().iloc[-1]
                ema_50 = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator().iloc[-1] # Fallback to SMA if EMA fails or use ta.trend.EMAIndicator
                ema_50 = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator().iloc[-1]
                
                results["indicators"]["EMA"] = {
                    "EMA_20": ema_20,
                    "EMA_50": ema_50,
                    "signal": self._get_ema_signal(df['Close'].iloc[-1], ema_20, ema_50)
                }
            
            # Stochastic
            if "STOCH" in indicators:
                stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
                k = stoch.stoch().iloc[-1]
                d = stoch.stoch_signal().iloc[-1]
                
                results["indicators"]["Stochastic"] = {
                    "K": k,
                    "D": d,
                    "signal": self._get_stoch_signal(k, d)
                }
            
            # 종합 신호
            results["overall_signal"] = self._get_overall_signal(results["indicators"])
            
            return results
            
        except Exception as e:
            logger.error(f"Technical analysis error for {ticker}: {e}")
            return {"error": str(e), "ticker": ticker}
    
    def _get_rsi_signal(self, rsi: float) -> str:
        """RSI 신호 해석"""
        if rsi > 70:
            return "Overbought - Consider selling"
        elif rsi < 30:
            return "Oversold - Consider buying"
        else:
            return "Neutral"
    
    def _get_macd_signal(self, macd: float, signal: float, histogram: float) -> str:
        """MACD 신호 해석"""
        if macd > signal and histogram > 0:
            return "Bullish - Buy signal"
        elif macd < signal and histogram < 0:
            return "Bearish - Sell signal"
        else:
            return "Neutral"
    
    def _get_bb_signal(self, price: float, upper: float, middle: float, lower: float) -> str:
        """볼린저 밴드 신호 해석"""
        if price > upper:
            return "Overbought - Price above upper band"
        elif price < lower:
            return "Oversold - Price below lower band"
        else:
            return "Neutral - Price within bands"
    
    def _get_ma_signal(self, price: float, sma_20: float, sma_50: float, sma_200: Optional[float]) -> str:
        """이동평균 신호 해석"""
        if price > sma_20 > sma_50:
            return "Bullish - Uptrend"
        elif price < sma_20 < sma_50:
            return "Bearish - Downtrend"
        else:
            return "Neutral"
    
    def _get_ema_signal(self, price: float, ema_20: float, ema_50: float) -> str:
        """EMA 신호 해석"""
        if ema_20 > ema_50 and price > ema_20:
            return "Strong Buy"
        elif ema_20 < ema_50 and price < ema_20:
            return "Strong Sell"
        else:
            return "Hold"
    
    def _get_stoch_signal(self, k: float, d: float) -> str:
        """스토캐스틱 신호 해석"""
        if k > 80 and d > 80:
            return "Overbought"
        elif k < 20 and d < 20:
            return "Oversold"
        elif k > d:
            return "Bullish crossover"
        elif k < d:
            return "Bearish crossover"
        else:
            return "Neutral"
    
    def _get_overall_signal(self, indicators: Dict) -> str:
        """종합 신호 계산"""
        buy_signals = 0
        sell_signals = 0
        
        for indicator, data in indicators.items():
            signal = data.get('signal', '')
            if 'buy' in signal.lower() or 'bullish' in signal.lower() or 'oversold' in signal.lower():
                buy_signals += 1
            elif 'sell' in signal.lower() or 'bearish' in signal.lower() or 'overbought' in signal.lower():
                sell_signals += 1
        
        if buy_signals > sell_signals * 1.5:
            return "STRONG BUY"
        elif buy_signals > sell_signals:
            return "BUY"
        elif sell_signals > buy_signals * 1.5:
            return "STRONG SELL"
        elif sell_signals > buy_signals:
            return "SELL"
        else:
            return "HOLD"

class ValuationMetricsInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    compare_to_industry: bool = Field(default=True, description="Compare to industry averages")

class ValuationMetricsTool(BaseTool):
    name = "valuation_metrics"
    description = "Calculate and analyze valuation metrics for fundamental analysis"
    args_schema = ValuationMetricsInput
    
    def _run(self, ticker: str, compare_to_industry: bool = True) -> Dict:
        """밸류에이션 지표 분석"""
        try:
            # FinanceDataReader does not support detailed financials/info
            # We will fetch price but return empty metrics to avoid errors
            
            start_date = get_start_date("1d")
            df = fdr.DataReader(ticker, start=start_date)
            
            if not df.empty:
                current_price = df['Close'].iloc[-1]
                market_cap = 0 # Not available in FDR
            else:
                current_price = 0
                market_cap = 0
            
            # 기본 밸류에이션 지표 (Stubbed)
            metrics = {
                "ticker": ticker,
                "current_price": float(current_price),
                "market_cap": market_cap,
                "enterprise_value": 0,
                "ratios": {
                    "PE_ratio": 0,
                    "Forward_PE": 0,
                    "PEG_ratio": 0,
                    "Price_to_Book": 0,
                    "Price_to_Sales": 0,
                    "EV_to_EBITDA": 0,
                    "EV_to_Revenue": 0
                },
                "profitability": {
                    "ROE": 0,
                    "ROA": 0,
                    "Profit_Margin": 0,
                    "Operating_Margin": 0,
                    "Gross_Margin": 0
                },
                "financial_health": {
                    "Current_Ratio": 0,
                    "Quick_Ratio": 0,
                    "Debt_to_Equity": 0,
                    "Interest_Coverage": 0,
                    "Free_Cash_Flow": 0
                },
                "growth": {
                    "Revenue_Growth": 0,
                    "Earnings_Growth": 0,
                    "Quarterly_Earnings_Growth": 0
                }
            }
            
            # 업종 평균과 비교
            if compare_to_industry:
                metrics["comparison"] = {
                    "sector": "Unknown",
                    "industry": "Unknown",
                    "valuation_assessment": "Unable to assess - Data unavailable"
                }
            
            # 종합 평가
            metrics["overall_assessment"] = {
                "score": "0/0",
                "percentage": 0,
                "recommendation": "HOLD",
                "key_points": ["Data unavailable for valuation"]
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Valuation analysis error for {ticker}: {e}")
            return {"error": str(e), "ticker": ticker}
    
    def _assess_valuation(self, ratios: Dict, sector: str) -> str:
        return "Unable to assess"
    
    def _get_overall_assessment(self, metrics: Dict) -> Dict:
        return {
            "score": "0/0",
            "percentage": 0,
            "recommendation": "HOLD",
            "key_points": ["Data unavailable"]
        }

class BacktestingInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    strategy: Dict = Field(description="Trading strategy parameters")
    period: str = Field(default="1y", description="Backtest period")

class BacktestingTool(BaseTool):
    name = "backtesting"
    description = "Backtest trading strategies with historical data"
    args_schema = BacktestingInput
    
    def _run(self, ticker: str, strategy: Dict, period: str = "1y") -> Dict:
        """전략 백테스팅"""
        try:
            # 데이터 가져오기
            start_date = get_start_date(period)
            df = fdr.DataReader(ticker, start=start_date)
            
            if df.empty:
                return {"error": "No data available for backtesting"}
            
            # 전략 파라미터
            buy_rsi = strategy.get("buy_rsi", 30)
            sell_rsi = strategy.get("sell_rsi", 70)
            stop_loss = strategy.get("stop_loss", 0.05)
            take_profit = strategy.get("take_profit", 0.15)
            
            # RSI 계산
            rsi = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            
            # 백테스팅 로직
            positions = []
            current_position = None
            capital = 100000  # 초기 자본
            
            for i in range(1, len(df)):
                current_price = df['Close'].iloc[i]
                current_rsi = rsi.iloc[i] if i < len(rsi) else 50
                
                # 매수 신호
                if current_position is None and current_rsi < buy_rsi:
                    current_position = {
                        "entry_date": df.index[i],
                        "entry_price": current_price,
                        "shares": capital / current_price,
                        "stop_loss": current_price * (1 - stop_loss),
                        "take_profit": current_price * (1 + take_profit)
                    }
                
                # 매도 신호
                elif current_position is not None:
                    should_sell = (
                        current_rsi > sell_rsi or
                        current_price <= current_position["stop_loss"] or
                        current_price >= current_position["take_profit"]
                    )
                    
                    if should_sell:
                        exit_reason = "RSI" if current_rsi > sell_rsi else \
                                     "Stop Loss" if current_price <= current_position["stop_loss"] else \
                                     "Take Profit"
                        
                        profit = (current_price - current_position["entry_price"]) * current_position["shares"]
                        profit_percentage = ((current_price - current_position["entry_price"]) / 
                                           current_position["entry_price"] * 100)
                        
                        positions.append({
                            "entry_date": current_position["entry_date"].strftime("%Y-%m-%d"),
                            "exit_date": df.index[i].strftime("%Y-%m-%d"),
                            "entry_price": round(current_position["entry_price"], 2),
                            "exit_price": round(current_price, 2),
                            "profit": round(profit, 2),
                            "profit_percentage": round(profit_percentage, 2),
                            "exit_reason": exit_reason
                        })
                        
                        capital += profit
                        current_position = None
            
            # 결과 계산
            total_trades = len(positions)
            winning_trades = len([p for p in positions if p["profit"] > 0])
            losing_trades = total_trades - winning_trades
            
            total_profit = sum(p["profit"] for p in positions)
            total_return = ((capital - 100000) / 100000 * 100)
            
            # Buy and Hold 비교
            buy_hold_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100)
            
            # 샤프 비율 계산
            if total_trades > 0:
                returns = [p["profit_percentage"] for p in positions]
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            return {
                "ticker": ticker,
                "period": period,
                "strategy_parameters": strategy,
                "performance": {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": round((winning_trades / total_trades * 100) if total_trades > 0 else 0, 2),
                    "total_profit": round(total_profit, 2),
                    "total_return": round(total_return, 2),
                    "final_capital": round(capital, 2),
                    "sharpe_ratio": round(sharpe_ratio, 2),
                    "buy_hold_return": round(buy_hold_return, 2),
                    "outperformance": round(total_return - buy_hold_return, 2)
                },
                "trades": positions[:10],  # 최근 10개 거래만
                "recommendation": self._get_backtest_recommendation(total_return, buy_hold_return, sharpe_ratio)
            }
            
        except Exception as e:
            logger.error(f"Backtesting error for {ticker}: {e}")
            return {"error": str(e), "ticker": ticker}
    
    def _get_backtest_recommendation(self, strategy_return: float, buy_hold_return: float, sharpe_ratio: float) -> str:
        """백테스트 결과 기반 추천"""
        if strategy_return > buy_hold_return * 1.5 and sharpe_ratio > 1:
            return "Excellent strategy - Significantly outperforms buy & hold"
        elif strategy_return > buy_hold_return and sharpe_ratio > 0.5:
            return "Good strategy - Outperforms buy & hold"
        elif strategy_return > 0:
            return "Acceptable strategy - Positive returns but consider improvements"
        else:
            return "Poor strategy - Consider different approach"