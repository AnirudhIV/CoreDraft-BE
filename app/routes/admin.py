from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.models import Document as DBDocument, User as DBUser
from app.auth.auth import get_current_user
from app.database.db import get_db

router = APIRouter()

# --- Permission check ---
def require_admin(user: DBUser):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
    return user

# View all users
@router.get("/users")
def list_users(
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)
):
    require_admin(current_user)
    return db.query(DBUser).all()

# Block / unblock user
@router.patch("/users/{user_id}/block")
def block_user(
    user_id: int,
    block: bool,
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)
):
    require_admin(current_user)
    
    user = db.query(DBUser).filter(DBUser.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prevent blocking an admin
    if user.role == "admin":
        raise HTTPException(status_code=403, detail="Cannot block an admin user")
    
    user.blocked = block  # Use the dedicated blocked field
    db.commit()
    db.refresh(user)
    
    return {"message": f"User {'blocked' if block else 'unblocked'}", "user": user.email}

# Promote / demote user
@router.patch("/users/{user_id}/role")
def change_role(
    user_id: int,
    role: str,
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)
):
    require_admin(current_user)
    if role not in ["user", "admin"]:
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'admin'")

    user = db.query(DBUser).filter(DBUser.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.role = role
    db.commit()
    db.refresh(user)
    return {"message": f"User role updated to {role}", "user": user.email}

# Reset password
@router.post("/users/{user_id}/reset-password")
def reset_password(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_user)
):
    require_admin(current_user)
    user = db.query(DBUser).filter(DBUser.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Example: reset to temp password "Temp123!"
    new_hashed_pw = "hashed-temp"  # replace with actual hash function
    user.hashed_password = new_hashed_pw
    db.commit()
    return {"message": f"Password reset for {user.email}. Temporary password issued."}

# Set default document
@router.put("/documents/{doc_id}/set-default")
def set_default_document(doc_id: int, db: Session = Depends(get_db), current_user: DBUser = Depends(get_current_user)):
    require_admin(current_user)

    # Reset all documents to not default
    db.query(DBDocument).update({DBDocument.is_default: False})
    db.commit()

    # Set the selected document as default
    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc.is_default = True
    db.commit()
    db.refresh(doc)

    return {"message": "Document set as default", "document_id": doc.id}
