from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSONB
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import relationship
import enum   



Base = declarative_base()

class RoleEnum(str, enum.Enum):
    admin = "admin"
    user = "user"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    documents = relationship("Document", back_populates="owner")
    
    # New role column (replaces is_admin)
    role = Column(Enum(RoleEnum), default=RoleEnum.user, nullable=False)
    blocked = Column(Boolean, default=False)


class Email(Base):
    __tablename__ = "emails"
    id = Column(Integer, primary_key=True, index=True)
    sender = Column(String)
    recipient = Column(String)
    subject = Column(String)
    body = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String, nullable=False)
    type = Column(String, default="document")              # e.g., policy, contract
    content = Column(Text, nullable=False)
    tags = Column(JSONB, default=[])                        # AI-suggested tags

    created_at = Column(DateTime, default=datetime.utcnow) # optional but helpful
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    owner = relationship("User", back_populates="documents")
    is_default = Column(Boolean, default=False)
