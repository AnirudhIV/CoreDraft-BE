# app/config.py

from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str
    class Config:
        env_file = ".env"

settings = Settings()

# You can use these directly elsewhere
GEMINI_API_KEY = settings.GEMINI_API_KEY
DATABASE_URL = settings.DATABASE_URL
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
