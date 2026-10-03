from typing import Literal
from pydantic import BaseModel, Field, model_validator

Pillar = Literal["Environmental", "Social", "Governance"]

class ESGKnowledgeUnit(BaseModel):
    pillar: Pillar
    topic: str = Field(min_length=2, max_length=120)
    knowledge_type: Literal["company_specific", "general_esg"]
    source_facts: list[str] = Field(min_length=1)
    extracted_knowledge: str = Field(min_length=20)
    source_supported: bool
    confidence: float = Field(ge=0.0, le=1.0)

class ESGExtractionResult(BaseModel):
    usable: bool
    rejection_reason: str | None = None
    knowledge_units: list[ESGKnowledgeUnit] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistency(self):
        if self.usable and not self.knowledge_units:
            raise ValueError("usable=true requires at least one knowledge unit")
        if not self.usable and self.knowledge_units:
            raise ValueError("usable=false requires an empty knowledge_units list")
        return self
