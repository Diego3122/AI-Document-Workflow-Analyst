from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.models.schemas import ReviewAction, TaskType


settings = get_settings()
API_BASE_URL = settings.ui_api_base_url.rstrip('/')
AUTH_MODE = settings.resolved_auth_mode()
PUBLIC_UI_MODE = settings.ui_public_enabled
TASK_TYPE_LABELS = {
    'auto': 'Let the app decide',
    TaskType.QUESTION_ANSWERING.value: 'Answer a question',
    TaskType.STRUCTURED_EXTRACTION.value: 'Pull out key details',
    TaskType.SUMMARIZATION.value: 'Summarize the document',
    TaskType.RISK_FLAGGING.value: 'Find possible risks',
}
REVIEW_ACTION_LABELS = {
    ReviewAction.APPROVE.value: 'Approve answer',
    ReviewAction.EDIT.value: 'Edit answer',
    ReviewAction.REJECT.value: 'Reject answer',
}
STATUS_LABELS = {
    'uploaded': 'Uploaded',
    'queued': 'Preparing',
    'indexed': 'Ready',
    'failed': 'Needs attention',
}


for key, default in {
    'documents': [],
    'selected_document_ids': [],
    'selected_prepare_document_id': '',
    'latest_upload_result': None,
    'latest_delete_result': None,
    'latest_prepare_result': None,
    'latest_query_result': None,
    'latest_review_result': None,
    'latest_metrics': None,
    'current_user': None,
    'current_auth_fingerprint': '',
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def request_context_headers() -> dict[str, str]:
    try:
        return {str(key).lower(): str(value) for key, value in st.context.headers.items()}
    except Exception:
        return {}


def extract_forwarded_access_token() -> str | None:
    headers = request_context_headers()
    for header_name in settings.ui_forwarded_access_token_headers:
        raw_value = headers.get(header_name.lower())
        if not raw_value:
            continue
        value = raw_value.strip()
        if not value:
            continue
        if value.lower().startswith('bearer '):
            token = value[7:].strip()
            if token:
                return token
            continue
        return value
    return None


def build_api_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if AUTH_MODE in {'jwt', 'hybrid'}:
        access_token = extract_forwarded_access_token()
        if access_token:
            headers['Authorization'] = f'Bearer {access_token}'
            return headers
    if AUTH_MODE in {'api_key', 'hybrid'} and settings.ui_api_key:
        headers['X-API-Key'] = settings.ui_api_key
    return headers


def auth_fingerprint(headers: dict[str, str]) -> str:
    if not headers:
        return ''
    payload = '|'.join(f'{key}:{value}' for key, value in sorted(headers.items()))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def api_get(path: str, *, headers: dict[str, str] | None = None) -> Any:
    request_headers = headers if headers is not None else build_api_headers()
    response = requests.get(f'{API_BASE_URL}{path}', timeout=15, headers=request_headers)
    response.raise_for_status()
    return response.json()


def api_post(path: str, *, json: dict | None = None, files: dict | None = None, headers: dict[str, str] | None = None) -> Any:
    request_headers = headers if headers is not None else build_api_headers()
    response = requests.post(f'{API_BASE_URL}{path}', json=json, files=files, timeout=30, headers=request_headers)
    response.raise_for_status()
    return response.json()


def api_delete(path: str, *, headers: dict[str, str] | None = None) -> Any:
    request_headers = headers if headers is not None else build_api_headers()
    response = requests.delete(f'{API_BASE_URL}{path}', timeout=30, headers=request_headers)
    response.raise_for_status()
    return response.json()


def format_api_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status_code = exc.response.status_code
        if PUBLIC_UI_MODE:
            if status_code >= 500:
                return 'The service hit an internal error. Please try again later.'
            if status_code in {401, 403}:
                return 'The service rejected this action.'
        try:
            payload = exc.response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict) and payload.get('detail'):
            return str(payload['detail'])
        if PUBLIC_UI_MODE:
            return 'The request could not be completed.'
    return str(exc)


def refresh_current_user() -> dict[str, Any] | None:
    if not settings.auth_is_enabled():
        st.session_state.current_user = None
        st.session_state.current_auth_fingerprint = ''
        return None

    headers = build_api_headers()
    fingerprint = auth_fingerprint(headers)
    if fingerprint != st.session_state.current_auth_fingerprint:
        st.session_state.current_user = None
        st.session_state.current_auth_fingerprint = fingerprint

    if st.session_state.current_user is None and headers:
        st.session_state.current_user = api_get('/auth/me', headers=headers)
    return st.session_state.current_user


def load_documents() -> None:
    try:
        st.session_state.documents = api_get('/documents')
    except Exception:
        pass


def humanize_status(status: str) -> str:
    return STATUS_LABELS.get(status, status.replace('_', ' ').title())


def humanize_task_type(task_type: str) -> str:
    return TASK_TYPE_LABELS.get(task_type, task_type.replace('_', ' ').title())


def humanize_note(note: str) -> str:
    replacements = {
        'Retrieved evidence was too weak to answer directly.': 'The app did not find enough strong supporting text to answer confidently.',
        'Synthesis style: insufficient_evidence': 'The answer was marked as incomplete because the source material was weak.',
        'The workflow marked this response as insufficiently supported by evidence.': 'The answer was held back because it was not well-supported by the document.',
        'Confidence fell below the normal auto-accept threshold.': 'Confidence was lower than the normal threshold for a trusted answer.',
    }
    return replacements.get(note, note)


def ready_documents() -> list[dict[str, Any]]:
    return [document for document in st.session_state.documents if document.get('status') == 'indexed']


def preparable_documents() -> list[dict[str, Any]]:
    return [
        document
        for document in st.session_state.documents
        if document.get('status') in {'uploaded', 'failed'}
    ]


def sync_selected_documents() -> None:
    ready_docs = ready_documents()
    ready_ids = {document['document_id'] for document in ready_docs}
    st.session_state.selected_document_ids = [
        document_id for document_id in st.session_state.selected_document_ids if document_id in ready_ids
    ]


def sync_prepare_selection() -> None:
    candidates = preparable_documents()
    candidate_ids = {document['document_id'] for document in candidates}
    current = st.session_state.selected_prepare_document_id
    if current in candidate_ids:
        return

    latest_upload = st.session_state.latest_upload_result or {}
    latest_upload_id = latest_upload.get('document_id')
    if latest_upload_id in candidate_ids:
        st.session_state.selected_prepare_document_id = latest_upload_id
    else:
        st.session_state.selected_prepare_document_id = ''


def document_name_map(documents: list[dict[str, Any]]) -> dict[str, str]:
    return {document['document_id']: document['filename'] for document in documents}


def render_documents(documents: list[dict[str, Any]]) -> None:
    if not documents:
        st.info('No files have been uploaded yet.')
        return

    rows = [
        {
            'File': document['filename'],
            'Status': humanize_status(document['status']),
            'Pages': document.get('page_count', 0),
            'Chunks': document.get('chunk_count', 0),
            'Notes': document.get('last_error') or ('Ready for questions' if document.get('status') == 'indexed' else 'Waiting to be prepared'),
        }
        for document in documents
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_query_result(result: dict[str, Any]) -> None:
    st.subheader('Answer')

    if result.get('needs_human_review'):
        st.warning('This answer may need a quick human check because the supporting evidence was limited.')
    else:
        st.success('This answer appears to be supported by the selected file.')

    st.write(result.get('answer', 'No answer returned.'))

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric('Confidence', f"{round(float(result.get('confidence', 0)) * 100)}%")
    metric_2.metric('Sources used', str(len(result.get('citations', []))))
    metric_3.metric('Response time', f"{float(result.get('latency_ms', 0)) / 1000:.1f}s")

    citations = result.get('citations', [])
    st.subheader('Sources')
    if citations:
        for index, citation in enumerate(citations, start=1):
            page_number = citation.get('page_number')
            page_text = f", page {page_number}" if page_number is not None else ''
            st.markdown(f"**Source {index}:** {citation.get('document_name', 'Document')}{page_text}")
            if citation.get('excerpt'):
                st.caption(citation['excerpt'])
    else:
        st.info('No strong source quote was available for this answer.')

    notes = result.get('validation_notes', [])
    if notes:
        with st.expander('Why this answer may need review'):
            for note in notes:
                st.write(f"- {humanize_note(note)}")


def render_metrics(metrics: dict[str, Any]) -> None:
    column_1, column_2, column_3, column_4 = st.columns(4)
    column_1.metric('Questions answered', str(metrics.get('total_requests', 0)))
    column_2.metric('Average confidence', f"{round(float(metrics.get('average_confidence', 0)) * 100)}%")
    column_3.metric('Answers needing review', f"{round(float(metrics.get('review_rate', 0)) * 100)}%")
    column_4.metric('Workflow errors', str(metrics.get('workflow_error_count', 0)))


def refresh_documents_state(*, selected_prepare_document_id: str | None = None, selected_document_ids: list[str] | None = None) -> None:
    load_documents()
    if selected_prepare_document_id is not None:
        st.session_state.selected_prepare_document_id = selected_prepare_document_id
    if selected_document_ids is not None:
        st.session_state.selected_document_ids = selected_document_ids
    sync_prepare_selection()
    sync_selected_documents()


st.set_page_config(
    page_title='AI Document Workflow Analyst',
    layout='centered',
    initial_sidebar_state='collapsed',
)

st.markdown(
    """
    <style>
        .stApp [data-testid=\"stAppViewContainer\"] > .main {
            max-width: 1080px;
            margin: 0 auto;
        }

        .stApp [data-testid=\"stAppViewBlockContainer\"] {
            max-width: 1080px;
            padding-top: 2rem;
            padding-left: 1.25rem;
            padding-right: 1.25rem;
        }

        .stApp a[href^=\"#\"] {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    settings.validate_public_ui_settings()
    settings.validate_production_settings()
except ValueError as exc:
    st.title('AI Document Workflow Analyst')
    st.error(str(exc))
    st.stop()

if PUBLIC_UI_MODE and API_BASE_URL.startswith('http://') and not API_BASE_URL.startswith(('http://localhost', 'http://127.0.0.1')):
    st.title('AI Document Workflow Analyst')
    st.error('Public UI mode requires an HTTPS backend URL unless the backend is local-only.')
    st.stop()

current_user: dict[str, Any] | None = None
if settings.auth_is_enabled():
    headers = build_api_headers()
    if AUTH_MODE == 'jwt' and 'Authorization' not in headers:
        st.title('AI Document Workflow Analyst')
        st.error(
            'This deployment requires a validated access token from the hosting platform. '
            'No forwarded bearer token was found in the current request.'
        )
        st.stop()
    if AUTH_MODE in {'api_key', 'hybrid'} and not headers:
        st.title('AI Document Workflow Analyst')
        st.error('This UI requires server-side credentials because backend authentication is enabled.')
        st.stop()
    try:
        current_user = refresh_current_user()
    except Exception as exc:
        st.title('AI Document Workflow Analyst')
        st.error(format_api_error(exc))
        st.stop()
    if current_user is None:
        st.title('AI Document Workflow Analyst')
        st.error('The current session could not be authenticated.')
        st.stop()

CAN_UPLOAD_AND_QUERY = bool(current_user.get('can_upload_and_query')) if current_user else True
CAN_REVIEW = bool(current_user.get('can_review')) if current_user else True
CAN_VIEW_METRICS = bool(current_user.get('can_view_metrics')) if current_user else True
if PUBLIC_UI_MODE:
    CAN_REVIEW = False
    CAN_VIEW_METRICS = False

if not st.session_state.documents and CAN_UPLOAD_AND_QUERY:
    load_documents()
sync_selected_documents()
sync_prepare_selection()

st.title('AI Document Workflow Analyst')
st.caption('A concise document tool: upload a file, prepare it for search, then ask a question.')

st.divider()

st.subheader('1. Upload')
st.write('Choose a PDF or TXT file. Uploading stores it in the workspace but does not make it searchable yet.')
if CAN_UPLOAD_AND_QUERY:
    upload = st.file_uploader('File to upload', type=['pdf', 'txt'])
    if st.button('Save uploaded file', use_container_width=True):
        if upload is None:
            st.warning('Choose a file first.')
        else:
            try:
                with st.spinner('Uploading your file...'):
                    upload_result = api_post(
                        '/documents/upload',
                        files={'file': (upload.name, upload.getvalue(), upload.type or 'application/octet-stream')},
                    )
                    st.session_state.latest_upload_result = upload_result
                    st.session_state.latest_prepare_result = None
                    refresh_documents_state(selected_prepare_document_id=upload_result['document_id'], selected_document_ids=[])
                st.success(f"{upload_result['filename']} was uploaded. Prepare it for search next.")
            except Exception as exc:
                st.error(format_api_error(exc))
else:
    st.info('Your signed-in account does not have permission to upload or query files.')

if st.session_state.latest_upload_result:
    upload_result = st.session_state.latest_upload_result
    st.caption(
        f"Latest upload: {upload_result['filename']} | "
        f"Status: {humanize_status(upload_result.get('status', 'uploaded'))}"
    )

st.divider()

st.subheader('2. Prepare for search')
st.write('Preparing parses the file, splits it into searchable chunks, and indexes it for retrieval.')
pending_documents = preparable_documents()
if not CAN_UPLOAD_AND_QUERY:
    st.info('You need upload access before you can prepare files.')
elif not pending_documents:
    st.info('Upload a file first. Files waiting to be prepared will appear here.')
else:
    prepare_ids = [document['document_id'] for document in pending_documents]
    prepare_options = [''] + prepare_ids
    current_prepare_document_id = st.session_state.selected_prepare_document_id
    if current_prepare_document_id not in prepare_options:
        current_prepare_document_id = ''
    selected_prepare_document_id = st.selectbox(
        'File to prepare',
        options=prepare_options,
        index=prepare_options.index(current_prepare_document_id),
        format_func=lambda document_id: (
            'Choose a file to prepare'
            if not document_id
            else next(
                document['filename']
                for document in pending_documents
                if document['document_id'] == document_id
            )
        ),
    )
    st.session_state.selected_prepare_document_id = selected_prepare_document_id
    if st.button('Prepare selected file', use_container_width=True):
        if not selected_prepare_document_id:
            st.warning('Choose a file to prepare.')
        else:
            try:
                with st.spinner('Preparing your file for search...'):
                    prepare_result = api_post('/documents/index', json={'document_id': selected_prepare_document_id})
                    st.session_state.latest_prepare_result = prepare_result
                    refresh_documents_state(
                        selected_prepare_document_id=selected_prepare_document_id,
                        selected_document_ids=[selected_prepare_document_id],
                    )
                st.success('The file is ready for questions.')
            except Exception as exc:
                st.error(format_api_error(exc))

if st.session_state.latest_prepare_result:
    prepare_result = st.session_state.latest_prepare_result
    st.caption(
        f"Latest prepared file: {prepare_result['document_id']} | Pages: {prepare_result.get('page_count', 0)} | "
        f"Chunks: {prepare_result.get('chunk_count', 0)}"
    )

st.subheader('Workspace files')
render_documents(st.session_state.documents)

if CAN_UPLOAD_AND_QUERY and st.session_state.documents:
    with st.expander('Manage files', expanded=False):
        file_name_lookup = document_name_map(st.session_state.documents)
        selected_delete_document_id = st.selectbox(
            'File to delete',
            options=[document['document_id'] for document in st.session_state.documents],
            format_func=lambda document_id: file_name_lookup.get(document_id, document_id),
        )
        if st.button('Delete selected file', use_container_width=True):
            try:
                with st.spinner('Deleting the selected file...'):
                    delete_result = api_delete(f'/documents/{selected_delete_document_id}')
                    st.session_state.latest_delete_result = delete_result
                    st.session_state.latest_prepare_result = None
                    st.session_state.latest_query_result = None
                    st.session_state.selected_document_ids = [
                        document_id
                        for document_id in st.session_state.selected_document_ids
                        if document_id != selected_delete_document_id
                    ]
                    if st.session_state.selected_prepare_document_id == selected_delete_document_id:
                        st.session_state.selected_prepare_document_id = ''
                    refresh_documents_state()
                st.success('The file was deleted.')
            except Exception as exc:
                st.error(format_api_error(exc))

if st.session_state.latest_delete_result:
    delete_result = st.session_state.latest_delete_result
    st.caption(f"Latest deleted file ID: {delete_result.get('document_id', '')}")

st.divider()

st.subheader('3. Ask a question')
st.write('Select one or more prepared files, ask a focused question, and review the answer with citations.')
available_documents = ready_documents()
available_document_map = document_name_map(available_documents)
if not CAN_UPLOAD_AND_QUERY:
    st.info('Your signed-in account does not have permission to query files.')
elif available_documents:
    selected_document_ids = st.multiselect(
        'Prepared files to search',
        options=[document['document_id'] for document in available_documents],
        default=st.session_state.selected_document_ids,
        format_func=lambda document_id: available_document_map.get(document_id, document_id),
        help='Choose one or more prepared files to search.',
    )
    st.session_state.selected_document_ids = selected_document_ids
    if not selected_document_ids:
        st.caption('No file is selected yet. Pick one or more prepared files when you are ready to ask a question.')
else:
    st.info('Prepare a file first so it becomes searchable.')

query_text = st.text_area(
    'Question',
    placeholder='What would you like to know about the selected file(s)?',
    height=120,
)
ask_controls_left, ask_controls_right = st.columns(2)
with ask_controls_left:
    task_type_hint = st.selectbox(
        'Answer style',
        options=list(TASK_TYPE_LABELS.keys()),
        format_func=lambda option: TASK_TYPE_LABELS[option],
        index=0,
        help='Leave this on "Let the app decide" unless you want a specific type of answer.',
    )
with ask_controls_right:
    top_k = st.slider(
        'Search breadth',
        min_value=1,
        max_value=10,
        value=5,
        help='Higher values search more sections before answering.',
    )

if st.button('Get answer', use_container_width=True):
    if not available_documents:
        st.warning('Prepare a file first.')
    elif not st.session_state.selected_document_ids:
        st.warning('Choose at least one prepared file.')
    elif len(query_text.strip()) < 3:
        st.warning('Please enter a longer question.')
    else:
        payload = {
            'query': query_text,
            'document_ids': st.session_state.selected_document_ids,
            'top_k': top_k,
        }
        if task_type_hint != 'auto':
            payload['task_type_hint'] = task_type_hint
        try:
            with st.spinner('Reading the selected file and preparing an answer...'):
                st.session_state.latest_query_result = api_post('/query', json=payload)
        except Exception as exc:
            st.error(format_api_error(exc))

if st.session_state.latest_query_result:
    st.caption(
        f"Latest answer type: {humanize_task_type(st.session_state.latest_query_result.get('task_type', 'auto'))}"
    )
    render_query_result(st.session_state.latest_query_result)

advanced_tools_available = CAN_REVIEW or CAN_VIEW_METRICS
if advanced_tools_available:
    st.divider()
    with st.expander('Advanced tools', expanded=False):
        if CAN_REVIEW:
            st.subheader('Review or correct an answer')
            latest_request_id = ''
            if st.session_state.latest_query_result:
                latest_request_id = st.session_state.latest_query_result.get('request_id', '')

            review_request_id = st.text_input('Request ID', value=latest_request_id)
            review_action = st.selectbox(
                'What do you want to do?',
                options=list(REVIEW_ACTION_LABELS.keys()),
                format_func=lambda option: REVIEW_ACTION_LABELS[option],
            )
            edited_answer = st.text_area('Edited answer', help='Only needed if you want to replace the answer.')
            reviewer_notes = st.text_area('Notes')
            if st.button('Save review', use_container_width=True):
                payload = {
                    'request_id': review_request_id,
                    'action': review_action,
                    'edited_answer': edited_answer or None,
                    'reviewer_notes': reviewer_notes or None,
                }
                try:
                    st.session_state.latest_review_result = api_post('/query/review', json=payload)
                    st.success('Review saved.')
                except Exception as exc:
                    st.error(format_api_error(exc))

        if CAN_VIEW_METRICS:
            st.subheader('App activity')
            if st.button('Refresh activity summary'):
                try:
                    st.session_state.latest_metrics = api_get('/metrics')
                except Exception as exc:
                    st.error(format_api_error(exc))

            if st.session_state.latest_metrics:
                render_metrics(st.session_state.latest_metrics)



