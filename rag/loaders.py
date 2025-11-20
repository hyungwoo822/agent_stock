from typing import List, Dict
import requests
from bs4 import BeautifulSoup
# import yfinance as yf (Removed)
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class YahooFinanceLoader:
    """Yahoo Finance 데이터 로더 (Replaced with FinanceDataReader)"""
    
    def load(self, tickers: List[str]) -> List[Dict]:
        """티커별 데이터 로드"""
        documents = []
        
        import FinanceDataReader as fdr
        from datetime import datetime, timedelta
        
        for ticker in tickers:
            try:
                # 회사 정보 문서 (Stubbed)
                company_doc = {
                    "content": f"""
                    Company: {ticker}
                    Sector: Unknown
                    Industry: Unknown
                    Market Cap: N/A
                    PE Ratio: N/A
                    Description: Data for {ticker} fetched via FinanceDataReader
                    """,
                    "metadata": {
                        "source": "finance_datareader",
                        "ticker": ticker,
                        "type": "company_info",
                        "timestamp": datetime.now().isoformat()
                    }
                }
                documents.append(company_doc)
                
                # 최근 가격 데이터
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                df = fdr.DataReader(ticker, start=start_date)
                
                if not df.empty:
                    price_doc = {
                        "content": f"""
                        Ticker: {ticker}
                        Current Price: ${df['Close'].iloc[-1]:.2f}
                        Month High: ${df['High'].max():.2f}
                        Month Low: ${df['Low'].min():.2f}
                        Volume: {df['Volume'].iloc[-1]:,.0f} if 'Volume' in df.columns else N/A
                        Price Change (Month): {((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100):.2f}%
                        """,
                        "metadata": {
                            "source": "finance_datareader",
                            "ticker": ticker,
                            "type": "price_data",
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    documents.append(price_doc)
                    
            except Exception as e:
                logger.error(f"Error loading data for {ticker}: {e}")
        
        return documents

class InvestingComLoader:
    """Investing.com 데이터 로더"""
    
    def load(self, urls: List[str]) -> List[Dict]:
        """URL별 데이터 로드"""
        documents = []
        
        for url in urls:
            try:
                response = requests.get(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 기사 제목과 내용 추출
                title = soup.find('h1', class_='articleHeader')
                content = soup.find('div', class_='articlePage')
                
                if title and content:
                    doc = {
                        "content": f"{title.text}\n\n{content.text[:1000]}",
                        "metadata": {
                            "source": "investing.com",
                            "url": url,
                            "type": "article",
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    documents.append(doc)
                    
            except Exception as e:
                logger.error(f"Error loading from {url}: {e}")
        
        return documents

class NewsAggregatorLoader:
    """뉴스 수집 로더"""
    
    def load(self, sources: Dict[str, List[str]]) -> List[Dict]:
        """다중 소스에서 뉴스 로드"""
        documents = []
        
        for source, urls in sources.items():
            for url in urls:
                try:
                    response = requests.get(url, timeout=10)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # 소스별 파싱 로직
                    if "bloomberg" in source.lower():
                        articles = self._parse_bloomberg(soup)
                    elif "reuters" in source.lower():
                        articles = self._parse_reuters(soup)
                    else:
                        articles = self._parse_generic(soup)
                    
                    for article in articles:
                        doc = {
                            "content": article["content"],
                            "metadata": {
                                "source": source,
                                "url": url,
                                "title": article.get("title", ""),
                                "type": "news",
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                        documents.append(doc)
                        
                except Exception as e:
                    logger.error(f"Error loading news from {url}: {e}")
        
        return documents
    
    def _parse_bloomberg(self, soup: BeautifulSoup) -> List[Dict]:
        """Bloomberg 파싱"""
        articles = []
        # Bloomberg specific parsing
        return articles
    
    def _parse_reuters(self, soup: BeautifulSoup) -> List[Dict]:
        """Reuters 파싱"""
        articles = []
        # Reuters specific parsing
        return articles
    
    def _parse_generic(self, soup: BeautifulSoup) -> List[Dict]:
        """일반 뉴스 사이트 파싱"""
        articles = []
        
        # 일반적인 article 태그 찾기
        for article in soup.find_all('article')[:5]:
            title = article.find(['h1', 'h2', 'h3'])
            content = article.find(['p', 'div'])
            
            if title and content:
                articles.append({
                    "title": title.text.strip(),
                    "content": content.text.strip()[:500]
                })
        
        return articles