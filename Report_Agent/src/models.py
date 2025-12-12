from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, HttpUrl


class Article(BaseModel):
    id: str
    source: str           # "BBC" or "GOV.UK"
    url: HttpUrl
    title: str
    published_at: Optional[datetime]
    raw_html: str
    clean_text: str


class Chunk(BaseModel):
    id: str
    article_id: str
    order: int
    text: str
    section: Optional[str] = None
    topic_label: Optional[str] = None
    created_at: datetime


class Report(BaseModel):
    id: str
    created_at: datetime
    topic: str
    summary: str
    takeaways: List[str]
    entities: Dict[str, List[str]]  # organisations, people, locations, terms


class ConversationTurn(BaseModel):
    timestamp: datetime
    user_question: str
    answer: str
    used_chunk_ids: List[str]
    used_report_ids: List[str]
