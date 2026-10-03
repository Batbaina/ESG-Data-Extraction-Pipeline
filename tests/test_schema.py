import pytest
from pydantic import ValidationError
from esg_pipeline.schemas.esg import ESGExtractionResult

def test_rejected_result():
    x = ESGExtractionResult(usable=False, rejection_reason="commercial context", knowledge_units=[])
    assert x.usable is False

def test_inconsistent_result_fails():
    with pytest.raises(ValidationError):
        ESGExtractionResult(usable=True, knowledge_units=[])
