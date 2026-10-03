from esg_pipeline.schemas.esg import ESGKnowledgeUnit
from esg_pipeline.quality.checks import check_unit

def test_grounded_unit_passes():
    source = "The company reported 100 tonnes of Scope 1 emissions in 2025."
    unit = ESGKnowledgeUnit(
        pillar="Environmental", topic="GHG emissions", knowledge_type="company_specific",
        source_facts=["The company reported 100 tonnes of Scope 1 emissions in 2025."],
        extracted_knowledge="The company reported 100 tonnes of Scope 1 emissions in 2025.",
        source_supported=True, confidence=0.95
    )
    assert check_unit(unit, source).passed
