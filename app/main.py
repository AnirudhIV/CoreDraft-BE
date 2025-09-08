from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

# Import your modules (make sure these exist)
try:
    from app.database import SessionLocal, engine, Base
    from app.database.models import Base, User, Email, Document
    from app.routes import users, emails, compliance, admin
    from app.auth import routes as auth_routes
    # from app.chroma.embedder import embed_text  # Comment out if causing issues
except ImportError as e:
    print(f"Import warning: {e}")
    # Create minimal fallback for testing
    pass

app = FastAPI(title="AI Compliance Backend", debug=False)  # Set debug=False for production

# CORS settings
origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://core-draft-fe-kdvu.vercel.app",  # Removed trailing slash
    "https://*.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create DB tables (only if database imports work)
try:
    Base.metadata.create_all(bind=engine)
except:
    print("Database initialization skipped")

# Health check route
@app.get("/")
def read_root():
    return {"message": "Compliance API is running", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "AI Compliance Backend"}

# Include your API route modules (wrap in try-except for safety)
try:
    app.include_router(users.router, prefix="/users", tags=["Users"])
    app.include_router(admin.router, prefix="/admin", tags=["Admin"]) 
    app.include_router(compliance.router, prefix="/compliance", tags=["Compliance"])
    app.include_router(auth_routes.router, prefix="/auth", tags=["Auth"])
except Exception as e:
    print(f"Router inclusion error: {e}")

# Vercel handler - this is crucial for Vercel deployment
handler = Mangum(app)

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)