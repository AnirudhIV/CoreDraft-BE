from fastapi import APIRouter, HTTPException
from app.models.policy import Policy
from app.config import supabase

router = APIRouter()

@router.post("/policies/")
def create_policy(policy: Policy):
    # Insert policy into Supabase
    result = supabase.table("policies").insert(
        policy.dict(exclude={"id", "created_at", "updated_at"})
    ).execute()

    if result.error:
        raise HTTPException(status_code=400, detail=result.error.message)

    return {"message": "Policy created successfully", "data": result.data}


@router.get("/policies/{user_id}")
def get_user_policies(user_id: str):
    # Fetch policies for a given user_id
    result = supabase.table("policies").select("*").eq("user_id", user_id).execute()

    if result.error:
        raise HTTPException(status_code=400, detail=result.error.message)

    return {"message": "Policies retrieved", "data": result.data}
