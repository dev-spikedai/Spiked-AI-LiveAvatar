from enum import Enum
from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
from uuid import UUID

RecrawlSchedule = Literal["NONE", "DAILY", "WEEKLY", "MONTHLY", "ONCE", None]

# --- General Models ---
class UploadResponse(BaseModel):
    message: str
    source_id: str
    filename: str

class CrawlResponse(BaseModel):
    message: str
    source_id: str
    url: str

class RecrawlRequest(BaseModel):
    schedule: Optional[RecrawlSchedule] = None

class StatusResponse(BaseModel):
    source_id: str = Field(..., alias="id")
    status: str = Field(..., alias="ingestion_status")
    progress: int = 0
    filename: Optional[str] = None
    url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str

    class Config:
        populate_by_name = True

class DocumentResponse(BaseModel):
    id: str
    filename: str
    url: str
    created_at: str
    description: Optional[str] = None
    spaces: Optional[List[str]] = Field(default_factory=list)
    status: str
    progress: int = 0
    recrawl_schedule: Optional[RecrawlSchedule] = None
    client_id: Optional[str] = None
    folder_id: Optional[str] = None
    file_size: Optional[int] = None  # bytes; null for websites / legacy rows

class UpdateSourceRequest(BaseModel):
    description: Optional[str] = None
    spaces: Optional[List[str]] = None
    client_id: Optional[str] = None
    folder_id: Optional[str] = None

class CrawlRequest(BaseModel):
    url: HttpUrl
    description: Optional[str] = None
    spaces: Optional[List[str]] = Field(default_factory=list)
    use_focus: bool = False
    client_id: Optional[str] = None
    folder_id: Optional[str] = None

class ChunkResponse(BaseModel):
    id: UUID
    source_id: UUID
    content: str
    created_at: datetime

    class Config:
        from_attributes = True

# --- Search AI Models ---
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    source_ids: Optional[List[str]] = None  # List of selected source IDs for scoped RAG
    # Optional per-request overlay selectors. When both are provided AND a matching
    # client_kyc_configs row exists, those KYC fields override user_configs in the
    # prompt. When omitted or unmatched, /ask/* uses user_configs (legacy behavior).
    client_id: Optional[str] = None
    kyc_id: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]

class AskBeyondResponse(BaseModel):
    answer: str
    question: str

class CompanyDomainsRequest(BaseModel):
    company_url: str = Field(..., min_length=1, max_length=500)

class CompanyDomainsResponse(BaseModel):
    product_domain: str
    sub_domains: str

class FollowupRequest(BaseModel):
    question: str
    context_hash: str

class FollowupQuestions(BaseModel):
    sales_followup: List[str] = Field(..., alias="salesFollowupQuestions")
    client_followup: List[str] = Field(..., alias="clientFollowupQuestions")

class FollowupResponse(BaseModel):
    follow_ups: FollowupQuestions = Field(..., alias="followUps")

# --- Settings Models ---
class SettingsModel(BaseModel):
    bot_name: str = Field(default="SpikedAI", alias="botName")
    selected_persona: str = Field(default="balanced", alias="selectedPersona")
    custom_prompt: Optional[str] = Field(default="", alias="customPrompt")
    selected_answer_styles: Optional[List[str]] = Field(default=[], alias="selectedAnswerStyles")
    meeting_domains: Optional[List[str]] = Field(default=["General Sales"], alias="meetingDomains")
    strategic_keywords: Optional[List[str]] = None
    executive_snapshot: Optional[str] = None
    seller_company: Optional[str] = None
    products_services: Optional[str] = None
    product_domain: Optional[str] = None
    client_company: Optional[str] = None
    seller_name: Optional[str] = None
    client_names: Optional[str] = None
    sub_domains: Optional[str] = None
    company_url: Optional[str] = None
    seller_linkedin_url: Optional[str] = None
    seller_job_profile: Optional[str] = None
    user_industry: Optional[str] = None

    # ✅ ADD THIS CONFIG: This stops Pydantic from dropping Supabase's snake_case data
    class Config:
        populate_by_name = True
        from_attributes = True

class MeetingGoalBase(BaseModel):
    goal_description: str = Field(..., json_schema_extra={"example": "Secure a follow-up meeting with the CTO."})
    evaluation_criteria: str = Field(..., json_schema_extra={"example": "A calendar invitation is sent and accepted."})
    emoji_icon: Optional[str] = Field(None, max_length=2, json_schema_extra={"example": "🎯"})

class MeetingGoalCreate(MeetingGoalBase):
    pass

class MeetingGoalUpdate(MeetingGoalBase):
    pass

class MeetingGoal(MeetingGoalBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# --- AI Training Models ---
class DocumentType(str, Enum):
    proposal = "proposal"
    technical = "technical"
    presentation = "presentation"
    general = "general"

class DifficultyLevel(str, Enum):
    easy = "Easy"
    medium = "Medium"

class Persona(str, Enum):
    technical_lead = "Technical Lead"
    business_manager = "Business Manager"
    c_suite_executive = "C-Suite Executive"

class AIMode(str, Enum):
    auto = "AI Auto"
    custom = "AI Custom"

class DocumentHeading(BaseModel):
    section: str
    title: str
    content_summary: str

class DocumentAnalysis(BaseModel):
    extractedTopics: List[str]
    keyFeatures: List[str]
    documentType: DocumentType
    extractedHeadings: List[DocumentHeading]

class AnalyzeDocumentRequest(BaseModel):
    content: str

class CompareResponse(BaseModel):
    score: int = Field(ge=0, le=10)
    coverage: str
    key_points_missed: List[str]
    feedback: str
    strengths: List[str]
    improvements: List[str]
    confidence_level: int = Field(ge=0, le=100)
    response_time: int

class CompareAnswersRequest(BaseModel):
    user_answer: str
    ideal_answer: str
    original_question: str
    response_time: int

class GeneratedQuestion(BaseModel):
    question: str
    difficulty_level: DifficultyLevel
    focus_area: str
    document_references: List[str]
    question_id: str

class GenerateQuestionsRequest(BaseModel):
    content: str
    document_references: List[str]
    persona: Persona
    objective: str
    max_questions: int = 15
    difficulty_levels: List[DifficultyLevel] = [DifficultyLevel.easy, DifficultyLevel.medium]
    ai_mode: AIMode = AIMode.auto
    custom_instructions: Optional[str] = None

class GenerateQuestionsResponse(BaseModel):
    questions: List[GeneratedQuestion]
    status: str
    document_count: int
    persona: str
    total_available: int

class IdealAnswerRequest(BaseModel):
    question: str
    content: str

class IdealAnswerResponse(BaseModel):
    answer: str

# --- Help Chat Models ---
class HelpChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = [] # List of {"role": "user"|"assistant", "content": "..."}

class HelpChatResponse(BaseModel):
    reply: str
    sources: List[str]

class IndexHelpDocsRequest(BaseModel):
    bucket_path: str = "help-docs" # Folder in supabase storage

# --- Client & Folder Models ---

class ClientCreate(BaseModel):
    name: str

class ClientUpdate(BaseModel):
    name: Optional[str] = None

class FolderCreate(BaseModel):
    name: str
    icon: Optional[str] = "folder"

class FolderUpdate(BaseModel):
    name: Optional[str] = None
    icon: Optional[str] = None

class FolderResponse(BaseModel):
    id: str
    client_id: str
    name: str
    icon: str
    sort_order: int
    is_default: bool
    document_count: int = 0

class ClientResponse(BaseModel):
    id: str
    name: str
    initials: str
    color: str
    document_count: int = 0
    created_at: datetime

class ClientTreeResponse(BaseModel):
    id: str
    name: str
    initials: str
    color: str
    document_count: int = 0
    total_size: int = 0  # sum of file_size (bytes) across this client's sources
    created_at: Optional[str] = None  # ISO timestamp, used for client-side sorting
    folders: List[FolderResponse] = []

class StorageUsageResponse(BaseModel):
    used_bytes: int
    limit_bytes: int

# --- Document Move Models ---

class MoveDocumentRequest(BaseModel):
    folder_id: str
    client_id: Optional[str] = None

class BulkMoveDocumentsRequest(BaseModel):
    source_ids: List[str]
    folder_id: str
    client_id: Optional[str] = None

# --- Document Clone Models ---

class CloneDocumentRequest(BaseModel):
    target_client_ids: List[str] = []
    clone_to_all: bool = False

# --- Product Extraction Models ---

class ExtractProductsRequest(BaseModel):
    client_id: str

class ExtractProductsResponse(BaseModel):
    client_id: str
    client_name: str
    products_services: List[str]
# --- Tutorial Chat Models ---
class AppContext(BaseModel):
    current_page: Optional[str] = None
    errors: Optional[List[str]] = []
    form_state: Optional[Dict[str, str]] = {}
    user_role: Optional[str] = None

class TutorialChatRequest(BaseModel):
    message: str
    context: str
    history: Optional[List[Dict[str, str]]] = []
    app_context: Optional[AppContext] = None

# --- Meeting Log Summary Models ---
class MeetingLogSummaryRequest(BaseModel):
    meeting_log_ids: List[str] = Field(..., min_length=1, json_schema_extra={"example": ["uuid-1", "uuid-2"]})
    question: Optional[str] = Field(None, json_schema_extra={"example": "What objections did the client raise?"})

class MeetingLogInfo(BaseModel):
    id: str
    title: Optional[str] = None
    meeting_started_at: Optional[str] = None

class MeetingLogSummaryResponse(BaseModel):
    summary: str
    meeting_logs: List[MeetingLogInfo]
    question: Optional[str] = None
