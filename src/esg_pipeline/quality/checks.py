import re
from dataclasses import dataclass, asdict
from ..schemas.esg import ESGKnowledgeUnit


@dataclass
class QCResult:
    passed: bool
    confidence_ok: bool
    source_supported_flag: bool
    facts_have_source_overlap: bool
    lexical_grounding_score: float
    self_contained: bool
    reasons: list[str]

    def to_dict(self):
        return asdict(self)


def _tokens(text: str):
    return set(re.findall(r"[A-Za-z0-9%]+", text.lower()))


def check_unit(
    unit: ESGKnowledgeUnit,
    source_text: str,
    min_confidence: float = 0.70,
    min_lexical_grounding: float = 0.45,
    require_source_supported: bool = True,
) -> QCResult:
    reasons = []

    confidence_ok = unit.confidence >= min_confidence
    if not confidence_ok:
        reasons.append("low_confidence")

    source_supported_flag = bool(unit.source_supported)
    if require_source_supported and not source_supported_flag:
        reasons.append("model_marked_not_source_supported")

    source_tokens = _tokens(source_text)
    fact_tokens = _tokens(" ".join(unit.source_facts))
    meaningful = {t for t in fact_tokens if len(t) > 3}
    overlap_ratio = len(meaningful & source_tokens) / max(1, len(meaningful))
    facts_have_source_overlap = overlap_ratio >= min_lexical_grounding
    if not facts_have_source_overlap:
        reasons.append("weak_lexical_grounding")

    knowledge = unit.extracted_knowledge.strip()
    self_contained = len(knowledge.split()) >= 8 and not knowledge.lower().startswith(("this ", "it ", "they "))
    if not self_contained:
        reasons.append("not_self_contained")

    passed = confidence_ok and facts_have_source_overlap and self_contained
    if require_source_supported:
        passed = passed and source_supported_flag

    return QCResult(
        passed=passed,
        confidence_ok=confidence_ok,
        source_supported_flag=source_supported_flag,
        facts_have_source_overlap=facts_have_source_overlap,
        lexical_grounding_score=round(overlap_ratio, 4),
        self_contained=self_contained,
        reasons=reasons,
    )
