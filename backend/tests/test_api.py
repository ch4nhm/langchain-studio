import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("app.api.router.process_qa_request")
def test_ask_endpoint(mock_process):
    mock_process.return_value = {
        "answer": "Mocked answer",
        "source_documents": [],
        "session_id": "test-session"
    }
    
    response = client.post("/ask", json={"query": "Test query?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Mocked answer"

@patch("app.api.router.process_sql_request")
def test_sql_endpoint(mock_process):
    mock_process.return_value = {
        "answer": "Mocked SQL answer",
        "sql_executed": "SELECT * FROM test;",
        "raw_result": "[]",
        "session_id": "test-session"
    }
    
    response = client.post("/sql", json={"query": "Count users"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Mocked SQL answer"
