"""Tests for Phase 2: PCAP flow replay and XAI faithfulness benchmarking."""

import io
from pathlib import Path
import pytest
from app import create_app
from app.pcap_utils import extract_flows_from_pcap
from app.model_utils import DATA_DIR, FEATURE_COLUMNS


@pytest.fixture(scope="module")
def app():
    """Create and configure an application instance for testing."""
    test_app = create_app()
    test_app.config.update({"TESTING": True})
    return test_app


@pytest.fixture(scope="module")
def client(app):
    """Test client for issuing HTTP requests."""
    return app.test_client()


def test_pcap_flow_extraction():
    """Verify that raw PCAP packets are converted into valid NSL-KDD schema rows."""
    pcap_path = DATA_DIR / "sample_traffic.pcap"
    assert pcap_path.is_file(), "sample_traffic.pcap must exist; run generate_sample_pcap.py"

    df_flows = extract_flows_from_pcap(pcap_path)
    assert not df_flows.empty
    assert list(df_flows.columns) == FEATURE_COLUMNS
    assert set(df_flows["protocol_type"].unique()).issubset({"tcp", "udp", "icmp"})


def test_pcap_upload_endpoint(client):
    """Verify /predict-pcap endpoint handles file upload and produces batch classification."""
    pcap_path = DATA_DIR / "sample_traffic.pcap"
    with open(pcap_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/predict-pcap",
        data={"pcap_file": (io.BytesIO(file_bytes), "sample_traffic.pcap")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    # Checks for the intrusion badge rendered directly in batch_result.html
    assert b"Intrusion" in response.data


def test_metrics_dashboard_includes_xai(client):
    """Verify metrics dashboard surfaces explainability faithfulness tests."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"Explainability Faithfulness Evaluation: Tree SHAP vs. LIME" in response.data
    assert b"Deletion AUC" in response.data
    assert b"Insertion AUC" in response.data