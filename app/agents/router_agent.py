from __future__ import annotations

from app.models.schemas import AgentRoute, TaskType


class RouterAgent:
    """Simple routing heuristic scaffold for a later LLM-backed router."""

    def route(self, query: str, task_type_hint: TaskType | None = None) -> AgentRoute:
        if task_type_hint is not None:
            return AgentRoute(
                task_type=task_type_hint,
                reason='Task type hint supplied by the caller.',
            )

        normalized = query.lower()
        if any(keyword in normalized for keyword in ('extract', 'field', 'json', 'invoice number')):
            return AgentRoute(
                task_type=TaskType.STRUCTURED_EXTRACTION,
                reason='The request looks like field extraction or structured output.',
            )
        if any(keyword in normalized for keyword in ('summarize', 'summary', 'overview')):
            return AgentRoute(
                task_type=TaskType.SUMMARIZATION,
                reason='The request looks like a document summarization task.',
            )
        if any(keyword in normalized for keyword in ('risk', 'flag', 'issue', 'exposure')):
            return AgentRoute(
                task_type=TaskType.RISK_FLAGGING,
                reason='The request looks like a risk or issue detection task.',
            )
        return AgentRoute(
            task_type=TaskType.QUESTION_ANSWERING,
            reason='The request looks like evidence-backed document question answering.',
        )
