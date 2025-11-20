import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import config
import logging

logger = logging.getLogger(__name__)

class InvestmentRAGSystem:
    """투자 전용 RAG 시스템"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # 벡터 스토어 초기화
        self.realtime_store = None  # 실시간 데이터
        self.knowledge_store = None  # 정적 지식
        
        self.last_update = {}
        self.update_interval = timedelta(hours=1)
        
    async def initialize(self):
        """RAG 시스템 초기화"""
        # 기본 지식 베이스 로드
        await self._load_knowledge_base()
        
        # 실시간 데이터 초기 로드
        await self._load_realtime_data()
        
        logger.info("RAG system initialized")
    
    async def _load_knowledge_base(self):
        """정적 지식 베이스 로드"""
        documents = []
        
        # PDF 문서 로드 (투자 서적, 리서치 페이퍼)
        pdf_paths = [
            "data/books/intelligent_investor.pdf",
            "data/books/security_analysis.pdf",
            "data/research/market_cycles.pdf"
        ]
        
        for pdf_path in pdf_paths:
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {pdf_path}: {e}")
        
        # 텍스트 분할
        if documents:
            splits = self.text_splitter.split_documents(documents)
            self.knowledge_store = FAISS.from_documents(splits, self.embeddings)
            logger.info(f"Loaded {len(splits)} knowledge base chunks")
    
    async def _load_realtime_data(self):
        """실시간 데이터 초기 로드"""
        from tools.market_tools import YahooFinanceTool, NewsAggregatorTool
        
        yahoo_tool = YahooFinanceTool()
        news_tool = NewsAggregatorTool()
        
        # 주요 지수 데이터
        indices = ["SPY", "QQQ", "DIA", "IWM"]
        realtime_docs = []
        
        for index in indices:
            try:
                # 가격 데이터
                price_data = yahoo_tool._run(index, "price", "1d")
                doc_content = f"Index: {index}\nPrice: ${price_data.get('current_price', 0)}\nChange: {price_data.get('change_percent', 0)}%"
                realtime_docs.append(doc_content)
                
                # 뉴스 데이터
                news_data = news_tool._run([index], limit=5)
                for article in news_data.get("articles", [])[:3]:
                    doc_content = f"News: {article.get('title', '')}\nSource: {article.get('source', '')}\nSentiment: {article.get('sentiment', '')}"
                    realtime_docs.append(doc_content)
                    
            except Exception as e:
                logger.error(f"Error loading realtime data for {index}: {e}")
        
        if realtime_docs:
            self.realtime_store = FAISS.from_texts(realtime_docs, self.embeddings)
            logger.info(f"Loaded {len(realtime_docs)} realtime data chunks")
    
    async def update_knowledge(self, human_feedback: Dict = None):
        """지식 베이스 업데이트 (Human-in-the-loop)"""
        current_time = datetime.now()
        
        # 업데이트 필요 여부 확인
        should_update = False
        for source, last_time in self.last_update.items():
            if current_time - last_time > self.update_interval:
                should_update = True
                break
        
        if not should_update and not human_feedback:
            return
        
        logger.info("Updating RAG knowledge base...")
        
        # 실시간 데이터 업데이트
        await self._update_realtime_data()
        
        # Human feedback 반영
        if human_feedback:
            await self._apply_human_feedback(human_feedback)
        
        # 오래된 데이터 정리
        self._cleanup_old_data()
        
        logger.info("RAG update completed")
    
    async def _update_realtime_data(self):
        """실시간 데이터 증분 업데이트"""
        from tools.market_tools import NewsAggregatorTool
        
        news_tool = NewsAggregatorTool()
        
        # 최신 뉴스만 가져오기
        keywords = ["stock market", "earnings", "federal reserve", "inflation"]
        news_data = news_tool._run(keywords, limit=20)
        
        new_docs = []
        for article in news_data.get("articles", []):
            # 중복 체크 (간단한 버전)
            doc_content = f"News: {article.get('title', '')}\nPublished: {article.get('published_at', '')}\nSentiment: {article.get('sentiment', '')}"
            new_docs.append(doc_content)
        
        if new_docs:
            # 기존 스토어에 추가
            if self.realtime_store:
                self.realtime_store.add_texts(new_docs)
            else:
                self.realtime_store = FAISS.from_texts(new_docs, self.embeddings)
            
            logger.info(f"Added {len(new_docs)} new documents to realtime store")
        
        self.last_update["realtime"] = datetime.now()
    
    async def _apply_human_feedback(self, feedback: Dict):
        """Human feedback 반영"""
        # 피드백 기반 문서 생성
        feedback_docs = []
        
        if "important_factors" in feedback:
            for factor in feedback["important_factors"]:
                doc = f"User Preference: Important factor - {factor}"
                feedback_docs.append(doc)
        
        if "avoid_sectors" in feedback:
            for sector in feedback["avoid_sectors"]:
                doc = f"User Preference: Avoid sector - {sector}"
                feedback_docs.append(doc)
        
        if "custom_rules" in feedback:
            for rule in feedback["custom_rules"]:
                doc = f"User Rule: {rule}"
                feedback_docs.append(doc)
        
        if feedback_docs:
            if self.knowledge_store:
                self.knowledge_store.add_texts(feedback_docs)
            else:
                self.knowledge_store = FAISS.from_texts(feedback_docs, self.embeddings)
            
            logger.info(f"Applied {len(feedback_docs)} human feedback documents")
    
    def _cleanup_old_data(self):
        """오래된 데이터 정리"""
        # 30일 이상 된 실시간 데이터 제거
        # (실제 구현에서는 메타데이터와 타임스탬프 필요)
        pass
    
    def search(self, query: str, k: int = 5, search_type: str = "all") -> List[Dict]:
        """벡터 검색"""
        results = []
        
        if search_type in ["all", "realtime"] and self.realtime_store:
            realtime_results = self.realtime_store.similarity_search_with_score(query, k=k)
            for doc, score in realtime_results:
                results.append({
                    "type": "realtime",
                    "content": doc.page_content,
                    "score": score,
                    "metadata": doc.metadata
                })
        
        if search_type in ["all", "knowledge"] and self.knowledge_store:
            knowledge_results = self.knowledge_store.similarity_search_with_score(query, k=k)
            for doc, score in knowledge_results:
                results.append({
                    "type": "knowledge",
                    "content": doc.page_content,
                    "score": score,
                    "metadata": doc.metadata
                })
        
        # 점수로 정렬
        results.sort(key=lambda x: x["score"], reverse=False)
        
        return results[:k]
    
    def get_context_for_decision(self, ticker: str, decision_type: str) -> str:
        """투자 결정을 위한 컨텍스트 생성"""
        # 관련 정보 검색
        query = f"{ticker} {decision_type} investment analysis"
        search_results = self.search(query, k=10)
        
        # 컨텍스트 구성
        context_parts = []
        
        # 실시간 정보 우선
        realtime_context = [r for r in search_results if r["type"] == "realtime"]
        if realtime_context:
            context_parts.append("Recent Market Information:")
            for result in realtime_context[:3]:
                context_parts.append(result["content"])
        
        # 지식 베이스 정보
        knowledge_context = [r for r in search_results if r["type"] == "knowledge"]
        if knowledge_context:
            context_parts.append("\nInvestment Principles:")
            for result in knowledge_context[:2]:
                context_parts.append(result["content"])
        
        return "\n".join(context_parts)
