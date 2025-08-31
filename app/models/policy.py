
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime

class Policy(BaseModel):
    id: Optional[str] = None
    user_id: str
    title: str
    content: str
    source_documents: Optional[List[Dict]] = []
    generated_by_ai: Optional[bool] = False
    version: Optional[str] = "1.0"
    status: Optional[str] = "draft"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
