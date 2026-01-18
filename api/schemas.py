from pydantic import BaseModel
from typing import Dict, Any


class ScoringRequest(BaseModel):
    features: Dict[str, Any]


class ScoringResponse(BaseModel):
    probability: float
    decision: str  # "Accordé" ou "Refusé"
    threshold_used: float