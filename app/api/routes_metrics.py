from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import require_access
from app.core.config import get_settings
from app.models.requests import EvalRunRequest
from app.models.responses import EvalRunResponse, MetricsResponse
from app.services.eval_service import EvalService
from app.services.logging_service import LoggingService
from app.services.security_service import AuthPrincipal


router = APIRouter(tags=['metrics'])


def resolve_dataset_path(dataset_path: str | None) -> Path | None:
    if not dataset_path:
        return None

    settings = get_settings()
    base_dir = settings.evals_dir.resolve()
    candidate = Path(dataset_path)
    resolved = (base_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError('Evaluation datasets must live inside the configured evals directory.') from exc
    if resolved.suffix.lower() != '.jsonl':
        raise ValueError('Evaluation datasets must use the .jsonl extension.')
    return resolved


@router.get('/metrics', response_model=MetricsResponse)
def get_metrics(
    _: AuthPrincipal | None = Depends(require_access('admin', required_scopes=('metrics.read',))),
) -> MetricsResponse:
    return LoggingService().fetch_metrics()


@router.post('/metrics/evaluate', response_model=EvalRunResponse)
def run_evaluation(
    request: EvalRunRequest,
    _: AuthPrincipal | None = Depends(require_access('admin', required_scopes=('metrics.write',))),
) -> EvalRunResponse:
    try:
        dataset_path = resolve_dataset_path(request.dataset_path)
        return EvalService().run(dataset_path=dataset_path, max_cases=request.max_cases)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get('/metrics/evaluations/latest', response_model=EvalRunResponse)
def get_latest_evaluation(
    _: AuthPrincipal | None = Depends(require_access('admin', required_scopes=('metrics.read',))),
) -> EvalRunResponse:
    evaluation = LoggingService().fetch_latest_evaluation()
    if evaluation is None:
        raise HTTPException(status_code=404, detail='No evaluation runs have been recorded yet.')
    return evaluation
