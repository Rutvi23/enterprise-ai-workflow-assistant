from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

SAMPLES = Path(__file__).resolve().parent.parent / "sample_docs"


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=str(tmp_path / "data"), llm_provider="mock", _env_file=None)


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
def svc(app):
    return app.state.services


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def vendor_pdf():
    return (SAMPLES / "vendor_agreement.pdf").read_bytes()


@pytest.fixture
def leave_pdf():
    return (SAMPLES / "leave_policy.pdf").read_bytes()


@pytest.fixture
def loaded(client, vendor_pdf, leave_pdf):
    for name, data in (("vendor_agreement.pdf", vendor_pdf), ("leave_policy.pdf", leave_pdf)):
        r = client.post("/api/documents", files={"file": (name, data, "application/pdf")})
        assert r.status_code == 201, r.text
    return client
