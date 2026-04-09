from __future__ import annotations

from time import perf_counter

from app.agents.retrieval_agent import RetrievalAgent
from app.agents.router_agent import RouterAgent
from app.agents.synthesis_agent import SynthesisAgent
from app.agents.validation_agent import ValidationAgent
from app.core.utils import generate_id
from app.models.requests import QueryRequest
from app.models.responses import QueryResponse
from app.models.schemas import ValidationReport


class QueryWorkflowService:
    """Runs the end-to-end query workflow without binding it to HTTP routes."""

    def __init__(self) -> None:
        self.router_agent = RouterAgent()
        self.retrieval_agent = RetrievalAgent()
        self.synthesis_agent = SynthesisAgent()
        self.validation_agent = ValidationAgent()

    def execute(self, request: QueryRequest, request_id: str | None = None) -> tuple[QueryResponse, ValidationReport]:
        workflow_request_id = request_id or generate_id('req')
        started_at = perf_counter()

        route = self.router_agent.route(request.query, request.task_type_hint)
        evidence = self.retrieval_agent.retrieve(
            query=request.query,
            top_k=request.top_k,
            task_type=route.task_type,
            document_ids=request.document_ids,
        )
        draft = self.synthesis_agent.synthesize(
            query=request.query,
            route=route,
            evidence=evidence,
        )
        validated = self.validation_agent.validate(draft=draft, evidence=evidence)
        latency_ms = round((perf_counter() - started_at) * 1000, 2)

        response = QueryResponse(
            request_id=workflow_request_id,
            task_type=validated.draft.task_type,
            answer=validated.draft.answer,
            citations=validated.draft.citations,
            confidence=validated.draft.confidence,
            needs_human_review=validated.needs_human_review,
            structured_data=validated.draft.structured_data,
            validation_notes=validated.report.notes,
            retrieved_chunk_ids=[chunk.chunk_id for chunk in evidence.retrieved_chunks],
            retrieval_stats=evidence.retrieval_stats,
            latency_ms=latency_ms,
        )
        return response, validated.report
