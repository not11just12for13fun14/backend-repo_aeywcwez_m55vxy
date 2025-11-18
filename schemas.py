"""
Database Schemas for the SaaS app

Collections:
- user: auth + billing
- project: logical grouping for a user's uploads and termbase
- job: processing job for a single input media
- term: per-project glossary mapping for consistent Indian→British/American conversions
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Literal, List, Dict, Any

class User(BaseModel):
    name: Optional[str] = Field(None, description="Full name")
    email: EmailStr = Field(..., description="Email address")
    password_hash: Optional[str] = Field(None, description="Hashed password")
    provider: Literal["password", "google", "github"] = Field("password")
    stripe_customer_id: Optional[str] = None
    plan: Literal["free", "pro", "enterprise"] = "free"
    credits: int = 0
    is_active: bool = True

class Project(BaseModel):
    user_id: str = Field(..., description="Owner user id")
    name: str = Field(...)
    description: Optional[str] = None

class Term(BaseModel):
    project_id: str = Field(...)
    source: str = Field(..., description="Indian English token/phrase")
    target_uk: Optional[str] = None
    target_us: Optional[str] = None
    note: Optional[str] = None

class Job(BaseModel):
    user_id: str
    project_id: Optional[str] = None
    filename: str
    status: Literal["queued", "processing", "completed", "failed"] = "queued"
    error: Optional[str] = None
    source_media_url: Optional[str] = None
    source_duration_sec: Optional[float] = None
    transcript_text: Optional[str] = None
    dialect_normalised_text: Optional[str] = None
    target_accent: Optional[Literal["british", "american"]] = None
    tts_audio_url: Optional[str] = None
    price_cents: Optional[int] = None
    currency: Literal["usd", "inr", "gbp"] = "usd"

# Flames viewer notes
# - Model name lowercased is collection name: user, project, term, job
