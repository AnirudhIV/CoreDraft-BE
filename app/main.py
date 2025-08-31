from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import SessionLocal, engine, Base
from app.database.models import Base, User, Email, Document
from app.routes import users, emails, compliance,admin
from app.chroma.embedder import embed_text
from app.auth import routes as auth_routes







# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Compliance Backend",debug=True)

# CORS (adjust as needed)
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check route
@app.get("/")
def read_root():
    return {"message": "Compliance API is running"}

# Include route files
app.include_router(users.router, prefix="/users", tags=["Users"])

app.include_router(admin.router, prefix="/admin", tags=["Admin"])
app.include_router(compliance.router, prefix="/compliance", tags=["Compliance"])
app.include_router(auth_routes.router, prefix="/auth", tags=["Auth"])
