"""
M42 CPG Report Evaluation UI — Streamlit app for testing the evaluation API.
"""

import json
import os

import requests
import streamlit as st
from azure.storage.blob import BlobServiceClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_METRICS = [
    "clinical_accuracy",
    "completeness",
    "safety_completeness",
    "relevance",
    "coherence",
    "evidence_traceability",
    "hallucination_score",
    "fih_detected",
]

METRIC_LABELS = {
    "clinical_accuracy": "Clinical Accuracy (1-5)",
    "completeness": "Completeness (1-5)",
    "safety_completeness": "Safety Completeness (1-5)",
    "relevance": "Relevance (1-5)",
    "coherence": "Coherence (1-5)",
    "evidence_traceability": "Evidence Traceability (1-5)",
    "hallucination_score": "Hallucination Score (1-4)",
    "fih_detected": "Factually Incorrect Hallucinations",
}

EVALUATION_MODELS = ["gpt-4o", "gpt-4o-mini"]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="M42 Evaluation UI", page_icon="🔬", layout="wide")
st.title("M42 CPG Report Evaluation")
st.caption("Test interface for the evaluation API")

# ---------------------------------------------------------------------------
# Sidebar — global settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    api_base_url = st.text_input(
        "API Base URL",
        value="http://localhost:8000",
        help="Base URL of the evaluation API (no trailing slash)",
    )

    st.subheader("Azure Storage")
    blob_connection_string = st.text_input(
        "Storage Connection String",
        value=os.environ.get("AZURE_STORAGE_CONNECTION_STRING", ""),
        type="password",
        help="Connection string with account key (from Azure Portal > Storage Account > Access keys)",
    )
    blob_json_container = st.text_input(
        "Report JSON Container",
        value="cpg-report-json",
        help="Container name where structured report JSONs are stored",
    )

    st.divider()
    st.subheader("Evaluation Config")
    evaluation_model = st.selectbox("Evaluation Model", EVALUATION_MODELS)
    selected_metrics = st.multiselect(
        "Metrics",
        options=ALL_METRICS,
        default=ALL_METRICS,
        format_func=lambda m: METRIC_LABELS.get(m, m),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_blob_service_client(conn_str: str) -> BlobServiceClient:
    """Create a BlobServiceClient from a connection string."""
    return BlobServiceClient.from_connection_string(conn_str)


@st.cache_data(ttl=300, show_spinner="Fetching blob list…")
def list_blob_files(conn_str: str, container_name: str) -> list[str]:
    """List JSON blobs in the given container."""
    client = _get_blob_service_client(conn_str)
    container = client.get_container_client(container_name)
    blobs = [b.name for b in container.list_blobs() if b.name.endswith(".json")]
    return sorted(blobs)


def fetch_blob_content(conn_str: str, container_name: str, blob_name: str) -> dict:
    """Download and parse a JSON blob."""
    client = _get_blob_service_client(conn_str)
    blob_client = client.get_blob_client(container_name, blob_name)
    data = blob_client.download_blob().readall()
    return json.loads(data)


def upload_document_to_blob(
    conn_str: str, container_name: str, blob_name: str, data: bytes
) -> str:
    """Upload a document to blob storage and return its URL."""
    client = _get_blob_service_client(conn_str)
    blob_client = client.get_blob_client(container_name, blob_name)
    blob_client.upload_blob(data, overwrite=True)
    return blob_client.url


def call_evaluate(url: str, payload: dict) -> dict:
    """POST to the evaluation endpoint and return the response."""
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


def render_full_doc_result(result: dict):
    """Render the full-document evaluation response."""
    st.subheader("Evaluation Result")

    col1, col2, col3 = st.columns(3)
    col1.metric("Report ID", result.get("report_id", "—"))
    col2.metric("Evaluation ID", result.get("evaluation_id", "—"))
    col3.metric("Model", result.get("evaluation_model", "—"))

    st.divider()

    # Metric scores
    score_metrics = [
        m for m in ALL_METRICS if m != "fih_detected" and result.get(m) is not None
    ]
    if score_metrics:
        cols = st.columns(min(len(score_metrics), 4))
        for i, m in enumerate(score_metrics):
            metric_data = result[m]
            score = metric_data.get("score") if isinstance(metric_data, dict) else None
            cols[i % len(cols)].metric(
                METRIC_LABELS.get(m, m),
                score if score is not None else "—",
            )

    # FIH details
    fih = result.get("fih_detected")
    if fih:
        st.subheader("Factually Incorrect Hallucinations")
        for item in fih:
            with st.expander(f"🔴 {item.get('severity', '').upper()} — {item.get('location', '')}"):
                st.markdown(f"**Claim:** {item.get('claim', '')}")
                st.markdown(f"**Source says:** {item.get('source_says', '')}")

    # Flags
    flags = result.get("flags", [])
    if flags:
        st.subheader("Flags")
        for f in flags:
            st.warning(f)

    # Raw JSON
    with st.expander("Raw JSON Response"):
        st.json(result)


def render_section_result(result: dict):
    """Render the section-wise evaluation response."""
    st.subheader("Section Evaluation Result")

    col1, col2, col3 = st.columns(3)
    col1.metric("Report ID", result.get("report_id", "—"))
    col2.metric("Evaluation ID", result.get("evaluation_id", "—"))
    col3.metric("Model", result.get("evaluation_model", "—"))

    st.divider()

    # Final aggregated scores
    final_scores = result.get("final_scores", {})
    if final_scores:
        st.subheader("Aggregated Scores")
        cols = st.columns(min(len(final_scores), 4))
        for i, (metric, score) in enumerate(final_scores.items()):
            cols[i % len(cols)].metric(METRIC_LABELS.get(metric, metric), f"{score:.2f}")

    # Per-section breakdown
    sections = result.get("section_scores", [])
    if sections:
        st.subheader("Per-Section Breakdown")
        for sec in sections:
            label = f"{sec.get('section_title', 'Untitled')} ({sec.get('section_type', '')})"
            with st.expander(label):
                for m in ALL_METRICS:
                    if m == "fih_detected":
                        continue
                    val = sec.get(m)
                    if val and isinstance(val, dict):
                        st.markdown(
                            f"**{METRIC_LABELS.get(m, m)}:** {val.get('score', '—')} "
                            f"(confidence: {val.get('confidence', '—')})"
                        )
                        if val.get("reasoning"):
                            st.caption(val["reasoning"])
                sec_flags = sec.get("flags", [])
                if sec_flags:
                    for f in sec_flags:
                        st.warning(f)

    # Global flags
    flags = result.get("flags", [])
    if flags:
        st.subheader("Flags")
        for f in flags:
            st.warning(f)

    with st.expander("Raw JSON Response"):
        st.json(result)


# ---------------------------------------------------------------------------
# Main content — tabs for the two evaluation modes
# ---------------------------------------------------------------------------

tab_section, tab_full = st.tabs(["Section-Wise Evaluation", "Full-Document Evaluation"])

# ===== Section-Wise Evaluation =====
with tab_section:
    st.subheader("Section-Wise Evaluation")

    guideline_topic = st.text_input(
        "Guideline Topic *",
        placeholder="e.g. First-line treatment for transplant-eligible NDMM",
        key="sec_topic",
    )
    disease_context = st.text_input(
        "Disease Context *",
        placeholder="e.g. Multiple Myeloma",
        key="sec_disease",
    )

    input_mode = st.radio(
        "Input Source",
        ["Upload JSON File", "Upload PDF/DOCX File", "Select from Azure Storage"],
        horizontal=True,
        key="sec_input_mode",
    )

    report_json_data = None
    uploaded_doc_bytes = None
    uploaded_doc_name = None

    if input_mode == "Upload JSON File":
        uploaded = st.file_uploader(
            "Upload Report JSON",
            type=["json"],
            key="sec_upload",
            help="Structured report JSON with report_id and sections array",
        )
        if uploaded:
            try:
                report_json_data = json.load(uploaded)
                st.success(
                    f"Loaded: {report_json_data.get('report_id', '?')} — "
                    f"{len(report_json_data.get('sections', []))} section(s)"
                )
            except json.JSONDecodeError:
                st.error("Invalid JSON file.")

    elif input_mode == "Upload PDF/DOCX File":
        uploaded_doc = st.file_uploader(
            "Upload PDF or DOCX Report",
            type=["pdf", "docx"],
            key="sec_doc_upload",
            help="Document will be processed via Azure Document Intelligence to extract sections",
        )
        if uploaded_doc:
            uploaded_doc_bytes = uploaded_doc.getvalue()
            uploaded_doc_name = uploaded_doc.name
            st.success(f"Loaded: {uploaded_doc_name} ({len(uploaded_doc_bytes) / 1024:.1f} KB)")

    else:  # Azure Storage
        if not blob_connection_string:
            st.info("Enter the Storage Connection String in the sidebar to browse files.")
        else:
            try:
                blobs = list_blob_files(blob_connection_string, blob_json_container)
                if blobs:
                    selected_blob = st.selectbox(
                        "Select Report JSON", blobs, key="sec_blob"
                    )
                else:
                    selected_blob = None
                    st.warning("No JSON files found in the container.")
            except Exception as e:
                selected_blob = None
                st.error(f"Failed to list blobs: {e}")

    # Submit
    if st.button("Run Section Evaluation", type="primary", key="sec_submit"):
        if not guideline_topic or not disease_context:
            st.error("Guideline Topic and Disease Context are required.")
        elif not selected_metrics:
            st.error("Select at least one metric.")
        else:
            payload: dict = {
                "guideline_topic": guideline_topic,
                "disease_context": disease_context,
                "metrics": selected_metrics,
                "evaluation_model": evaluation_model,
            }

            if input_mode == "Upload JSON File":
                if report_json_data is None:
                    st.error("Please upload a JSON file first.")
                    st.stop()
                payload["report_json"] = report_json_data
            elif input_mode == "Upload PDF/DOCX File":
                if uploaded_doc_bytes is None:
                    st.error("Please upload a PDF or DOCX file first.")
                    st.stop()
                if not blob_connection_string:
                    st.error("Storage Connection String is required in the sidebar to upload documents.")
                    st.stop()
                # Upload document to blob storage, then pass the URL as file_path
                doc_blob_url = upload_document_to_blob(
                    blob_connection_string, "documents", uploaded_doc_name, uploaded_doc_bytes
                )
                payload["file_path"] = doc_blob_url
            else:
                if not selected_blob:
                    st.error("Please select a blob file.")
                    st.stop()
                payload["json_path"] = f"{blob_json_container}/{selected_blob}"

            url = f"{api_base_url.rstrip('/')}/api/v1/evaluate/sections"
            with st.spinner("Running section evaluation…"):
                try:
                    result = call_evaluate(url, payload)
                    render_section_result(result)
                except requests.HTTPError as e:
                    st.error(f"API error {e.response.status_code}: {e.response.text}")
                except requests.ConnectionError:
                    st.error(f"Cannot connect to {url}. Is the API running?")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")


# ===== Full-Document Evaluation =====
with tab_full:
    st.subheader("Full-Document Evaluation")

    report_id = st.text_input(
        "Report ID *",
        placeholder="e.g. rpt-20250319-ndmm-001",
        key="full_report_id",
    )
    full_guideline_topic = st.text_input(
        "Guideline Topic *",
        placeholder="e.g. First-line treatment for transplant-eligible NDMM",
        key="full_topic",
    )
    full_disease_context = st.text_input(
        "Disease Context *",
        placeholder="e.g. Multiple Myeloma",
        key="full_disease",
    )
    generated_report = st.text_area(
        "Generated Report *",
        height=300,
        placeholder="Paste the full text of the generated CPG report here…",
        key="full_report_text",
    )

    if st.button("Run Full-Document Evaluation", type="primary", key="full_submit"):
        if not all([report_id, full_guideline_topic, full_disease_context, generated_report]):
            st.error("All fields are required.")
        elif not selected_metrics:
            st.error("Select at least one metric.")
        else:
            payload = {
                "report_id": report_id,
                "generated_report": generated_report,
                "guideline_topic": full_guideline_topic,
                "disease_context": full_disease_context,
                "metrics": selected_metrics,
                "evaluation_model": evaluation_model,
            }

            url = f"{api_base_url.rstrip('/')}/api/v1/evaluate"
            with st.spinner("Running full-document evaluation…"):
                try:
                    result = call_evaluate(url, payload)
                    render_full_doc_result(result)
                except requests.HTTPError as e:
                    st.error(f"API error {e.response.status_code}: {e.response.text}")
                except requests.ConnectionError:
                    st.error(f"Cannot connect to {url}. Is the API running?")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
