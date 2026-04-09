from __future__ import annotations

import pytest

from app.models.requests import QueryRequest, ReviewDecisionRequest



def test_query_request_sanitizes_and_deduplicates_document_ids() -> None:
    request = QueryRequest(
        query='  What\x00 are   the payment terms?  ',
        document_ids=['doc_abc123def456', 'doc_abc123def456'],
        metadata_filters={' team ': ' finance\x00dept '},
    )

    assert request.query == 'What are the payment terms?'
    assert request.document_ids == ['doc_abc123def456']
    assert request.metadata_filters == {'team': 'finance dept'}



def test_review_request_rejects_invalid_request_id() -> None:
    with pytest.raises(ValueError):
        ReviewDecisionRequest(request_id='bad-id', action='approve')
