from pydantic import BaseModel, EmailStr,Field
from typing import Optional, List


# -------------------------------
# User Schemas
# -------------------------------

class UserBase(BaseModel):
    email: EmailStr





class User(UserBase):
    id: int

    class Config:
        orm_mode = True


# -------------------------------
# Email Schemas
# -------------------------------

class EmailCreate(BaseModel):
    to: str
    subject: str
    body: str

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, String


class GeneratePromptRequest(BaseModel):

    prompt: str
    type: str

# -------------------------------
# Document Schemas
# -------------------------------


class DocumentCreate(BaseModel):
    title: str
    type: str
    content: str
    tags: Optional[List[str]] = []  # ← Add this

class DocumentUpdate(BaseModel):
    title: Optional[str]
    type: Optional[str]
    content: Optional[str]
    tags: Optional[List[str]]

class DocumentOut(BaseModel):
    id: int
    title: str
    type: str
    content: str
    tags: List[str] = Field(default_factory=list)
    is_default: bool
    user_id: int

class PromptInput(BaseModel):
    prompt: str

class ContentInput(BaseModel):
    doc_id: Optional[int] = None
    content: Optional[str] = None
class QuestionInput(BaseModel):
    question: str

class Config:
    orm_mode = True