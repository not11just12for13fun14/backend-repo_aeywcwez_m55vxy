import os
import io
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Literal, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from jose import jwt, JWTError
from passlib.context import CryptContext

from database import db, create_document
from schemas import User as UserSchema

# Optional external services
import stripe
from openai import OpenAI

# Media processing
from moviepy.editor import VideoFileClip, AudioFileClip

# ---------------------------
# Environment & Config
# ---------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALGO = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")
stripe.api_key = STRIPE_API_KEY if STRIPE_API_KEY else None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FILE_BYTES = 1_000_000_000  # 1GB

# ---------------------------
# App Init
# ---------------------------
app = FastAPI(title="Dialect Convert SaaS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------
# Helpers
# ---------------------------

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGO)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def get_collection(name: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    return db[name]


async def get_current_user(token: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token.credentials, JWT_SECRET, algorithms=[JWT_ALGO])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_collection("user").find_one({"_id": uuid.UUID(user_id) if len(user_id) == 36 else user_id})
    if not user:
        # Fallback by email if sub is email
        user = get_collection("user").find_one({"email": user_id})
    if not user:
        raise credentials_exception
    # normalize id
    user["id"] = str(user.get("_id"))
    return user


# ---------------------------
# Models
# ---------------------------
class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None

class TermPayload(BaseModel):
    source: str
    target_uk: Optional[str] = None
    target_us: Optional[str] = None
    note: Optional[str] = None

class JobResponse(BaseModel):
    job_id: str
    status: str
    transcript_text: Optional[str] = None
    dialect_normalised_text: Optional[str] = None
    tts_audio_url: Optional[str] = None

# ---------------------------
# Auth Routes
# ---------------------------
@app.post("/auth/signup")
def signup(payload: SignupRequest):
    users = get_collection("user")
    existing = users.find_one({"email": payload.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    stripe_customer_id = None
    if stripe.api_key:
        try:
            customer = stripe.Customer.create(email=payload.email, name=payload.name or payload.email)
            stripe_customer_id = customer["id"]
        except Exception:
            # Continue without Stripe
            stripe_customer_id = None

    user_doc = {
        "name": payload.name,
        "email": payload.email,
        "password_hash": get_password_hash(payload.password),
        "provider": "password",
        "stripe_customer_id": stripe_customer_id,
        "plan": "free",
        "credits": 0,
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    inserted_id = users.insert_one(user_doc).inserted_id

    token = create_access_token({"sub": payload.email})
    return {"token": token, "user": {"email": payload.email, "name": payload.name, "plan": "free"}}


@app.post("/auth/login")
def login(payload: LoginRequest):
    users = get_collection("user")
    user = users.find_one({"email": payload.email})
    if not user or not verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": payload.email})
    return {"token": token, "user": {"email": user["email"], "name": user.get("name"), "plan": user.get("plan", "free")}}


# ---------------------------
# Projects & Terms
# ---------------------------
@app.post("/projects")
def create_project(body: ProjectCreate, current_user: dict = Depends(get_current_user)):
    projects = get_collection("project")
    doc = {
        "user_id": current_user.get("_id", current_user.get("id", current_user.get("email"))),
        "name": body.name,
        "description": body.description,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    pid = projects.insert_one(doc).inserted_id
    return {"id": str(pid), "name": body.name}


@app.get("/projects")
def list_projects(current_user: dict = Depends(get_current_user)):
    projects = list(get_collection("project").find({"user_id": current_user.get("_id", current_user.get("id", current_user.get("email")))}))
    for p in projects:
        p["id"] = str(p.pop("_id"))
    return {"projects": projects}


@app.post("/projects/{project_id}/terms")
def upsert_terms(project_id: str, terms: List[TermPayload], current_user: dict = Depends(get_current_user)):
    col = get_collection("term")
    for t in terms:
        col.update_one(
            {"project_id": project_id, "source": t.source},
            {"$set": {"target_uk": t.target_uk, "target_us": t.target_us, "note": t.note}},
            upsert=True,
        )
    return {"ok": True}


@app.get("/projects/{project_id}/terms")
def get_terms(project_id: str, current_user: dict = Depends(get_current_user)):
    col = get_collection("term")
    terms = list(col.find({"project_id": project_id}))
    for t in terms:
        t["id"] = str(t.pop("_id"))
    return {"terms": terms}

# ---------------------------
# Billing (Stripe)
# ---------------------------
@app.post("/billing/checkout")
def create_checkout_session(request: Request, current_user: dict = Depends(get_current_user)):
    if not stripe.api_key:
        raise HTTPException(status_code=400, detail="Stripe not configured")
    origin = request.headers.get("origin") or request.headers.get("referer") or "http://localhost:3000"
    try:
        session = stripe.checkout.Session.create(
            customer=current_user.get("stripe_customer_id"),
            line_items=[{"price": os.getenv("STRIPE_PRICE_ID", "price_123"), "quantity": 1}],
            mode="subscription",
            success_url=origin + "?success=true",
            cancel_url=origin + "?canceled=true",
        )
        return {"url": session.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/billing/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    event = None
    try:
        if endpoint_secret:
            event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        else:
            event = stripe.Event.construct_from(payload, stripe.api_key)
    except Exception:
        return {"received": True}

    if event and event.get("type") == "checkout.session.completed":
        session = event["data"]["object"]
        # handle subscription activation if needed
    return {"received": True}

# ---------------------------
# Core: Upload, Transcribe, Convert, TTS
# ---------------------------

def is_video_filename(name: str) -> bool:
    ext = os.path.splitext(name.lower())[1]
    return ext in [".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"]


def is_audio_filename(name: str) -> bool:
    ext = os.path.splitext(name.lower())[1]
    return ext in [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"]


def extract_audio(input_path: str, output_path: str) -> None:
    clip = VideoFileClip(input_path)
    if not clip.audio:
        raise HTTPException(status_code=400, detail="No audio track found in video")
    clip.audio.write_audiofile(output_path, verbose=False, logger=None)
    clip.close()


def transcribe_audio(path: str) -> str:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI not configured for transcription")
    with open(path, "rb") as f:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )
    # transcript.text contains the full text
    return transcript.text


INDIAN_TO_BRITISH = {
    "prepone": "bring forward",
    "out of station": "away",
    "passed out": "graduated",
    "revert back": "reply",
    "co-brother": "brother-in-law",
    "kindly do the needful": "please handle it",
}

SPELLING_UK = {
    "color": "colour",
    "organize": "organise",
    "organization": "organisation",
    "center": "centre",
    "meter": "metre",
}

SPELLING_US = {
    "colour": "color",
    "organise": "organize",
    "organisation": "organization",
    "centre": "center",
    "metre": "meter",
}


def apply_rules(text: str, target: Literal["british", "american"], glossary: List[Dict]) -> str:
    # Phrase-level replacements
    for src, dst in INDIAN_TO_BRITISH.items():
        if target == "british":
            text = text.replace(src, dst)
        else:
            # for US, prefer simpler American phrasing (use same dst for now)
            text = text.replace(src, dst)

    # Project glossary
    for term in glossary:
        if target == "british" and term.get("target_uk"):
            text = text.replace(term["source"], term["target_uk"])
        if target == "american" and term.get("target_us"):
            text = text.replace(term["source"], term["target_us"])

    # Spelling normalisation
    if target == "british":
        for a, b in SPELLING_UK.items():
            text = text.replace(a, b)
    else:
        for a, b in SPELLING_US.items():
            text = text.replace(a, b)

    return text


def refine_with_llm(text: str, target: Literal["british", "american"], glossary: List[Dict]) -> str:
    if not openai_client:
        return text
    system = (
        "You rewrite transcripts to the target dialect while preserving meaning and context. "
        "Maintain consistency: always use the same term for the same concept within a single text. "
        "Only change dialect-specific phrasing and spelling, not factual content."
    )
    glossary_lines = []
    for t in glossary:
        if target == "british" and t.get("target_uk"):
            glossary_lines.append(f"{t['source']} -> {t['target_uk']}")
        if target == "american" and t.get("target_us"):
            glossary_lines.append(f"{t['source']} -> {t['target_us']}")
    glossary_block = "\n".join(glossary_lines)

    prompt = (
        f"Target dialect: {'British English' if target=='british' else 'American English'}\n"
        f"Glossary (enforce if present):\n{glossary_block}\n\n"
        f"Text:\n{text}\n\nRewrite the full text in the target dialect, returning only the rewritten text."
    )
    resp = openai_client.chat.completions.create(
        model=os.getenv("OPENAI_DIALECT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def synthesize_tts(text: str, target: Literal["british", "american"], out_path: str) -> None:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI not configured for TTS")
    voice = os.getenv("TTS_VOICE_US", "alloy") if target == "american" else os.getenv("TTS_VOICE_UK", "verse")
    # tts-1 supports mp3 output
    audio = openai_client.audio.speech.create(
        model=os.getenv("OPENAI_TTS_MODEL", "tts-1"),
        voice=voice,
        input=text,
        format="mp3",
    )
    with open(out_path, "wb") as f:
        f.write(audio.content)


@app.post("/jobs/upload", response_model=JobResponse)
async def upload_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_accent: Literal["british", "american"] = Form("british"),
    project_id: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user),
):
    # Basic size guard (Content-Length if present)
    length = None
    try:
        length = int(file.headers.get("content-length") or 0)
    except Exception:
        length = 0
    if length and length > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 1GB)")

    job_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(input_path, "wb") as f:
        data = await file.read()
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail="File too large (max 1GB)")
        f.write(data)

    # Prepare job record
    jobs = get_collection("job")
    job_doc = {
        "_id": job_id,
        "user_id": current_user.get("_id", current_user.get("id", current_user.get("email"))),
        "project_id": project_id,
        "filename": file.filename,
        "status": "processing",
        "target_accent": target_accent,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    jobs.insert_one(job_doc)

    # Process synchronously for MVP
    try:
        # 1) If video, extract audio to wav
        audio_path = input_path
        if is_video_filename(file.filename):
            audio_path = os.path.join(OUTPUT_DIR, f"{job_id}.wav")
            extract_audio(input_path, audio_path)
        elif not is_audio_filename(file.filename):
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # 2) Transcribe
        transcript_text = transcribe_audio(audio_path)

        # 3) Load glossary for project
        glossary = []
        if project_id:
            glossary = list(get_collection("term").find({"project_id": project_id}))

        # 4) Rule-based normalisation
        normalised = apply_rules(transcript_text, target_accent, glossary)
        # 5) Refine with LLM
        normalised = refine_with_llm(normalised, target_accent, glossary)

        # 6) TTS
        tts_out = os.path.join(OUTPUT_DIR, f"{job_id}.mp3")
        synthesize_tts(normalised, target_accent, tts_out)

        audio_url = f"/outputs/{job_id}.mp3"

        jobs.update_one(
            {"_id": job_id},
            {"$set": {
                "status": "completed",
                "transcript_text": transcript_text,
                "dialect_normalised_text": normalised,
                "tts_audio_url": audio_url,
                "updated_at": datetime.now(timezone.utc),
            }}
        )
        return JobResponse(job_id=job_id, status="completed", transcript_text=transcript_text, dialect_normalised_text=normalised, tts_audio_url=audio_url)
    except HTTPException as he:
        jobs.update_one({"_id": job_id}, {"$set": {"status": "failed", "error": he.detail}})
        raise
    except Exception as e:
        jobs.update_one({"_id": job_id}, {"$set": {"status": "failed", "error": str(e)}})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, current_user: dict = Depends(get_current_user)):
    job = get_collection("job").find_one({"_id": job_id, "user_id": current_user.get("_id", current_user.get("id", current_user.get("email")))})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(
        job_id=job["_id"],
        status=job["status"],
        transcript_text=job.get("transcript_text"),
        dialect_normalised_text=job.get("dialect_normalised_text"),
        tts_audio_url=job.get("tts_audio_url"),
    )


@app.get("/")
def root():
    return {"message": "Dialect Convert SaaS API running"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:50]}"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    return response

# Static files: serve outputs directory
from fastapi.staticfiles import StaticFiles
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
