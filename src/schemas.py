from typing import List, Literal
from pydantic import BaseModel, Field

class AnomalyDetail(BaseModel):
    label: str = Field(description="Name of the detected object, feature, or defect")
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Assessed severity level"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
    )
    observation: str = Field(description="Detailed physical description of the region")
    suggested_action: str = Field(description="Prescribed action or observation note")

class InspectionReport(BaseModel):
    summary: str = Field(description="Executive summary of the visual inspection")
    overall_status: Literal["PASS", "REVIEW_REQUIRED", "FAIL"]
    findings: List[AnomalyDetail]