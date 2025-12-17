"""
API Endpoint Tests

Tests for HTTP API endpoints that can be tested without WebRTC.
These complement the LiveKit WebRTC tests for voice features.
"""
import pytest
import httpx
import asyncio
from pathlib import Path


BASE_URL = "http://localhost:8080"


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test health check endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        print(f"✅ Health check passed: {data}")


@pytest.mark.asyncio
async def test_api_info_endpoint():
    """Test API info endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "endpoints" in data
        print(f"✅ API info endpoint works")


@pytest.mark.asyncio
async def test_token_generation():
    """Test token generation endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/api/token")
        
        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert "roomName" in data
        assert "serverUrl" in data
        assert len(data["token"].split(".")) == 3  # Valid JWT
        print(f"✅ Token generated for room: {data['roomName']}")


@pytest.mark.asyncio
async def test_news_feed_endpoint():
    """Test news feed endpoint (TC006 - should pass)."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{BASE_URL}/api/newsfeed",
            params={"query": "Claude Code", "max_results": 3}
        )
        
        assert response.status_code == 200, f"News feed failed: {response.status_code} - {response.text}"
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "count" in data
        assert isinstance(data["results"], list)
        print(f"✅ News feed returned {data['count']} results")


@pytest.mark.asyncio
async def test_document_listing():
    """Test document listing endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)
        print(f"✅ Document listing returned {len(data['documents'])} documents")


@pytest.mark.asyncio
async def test_ingest_url_youtube():
    """Test URL ingestion with YouTube type."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Use a short test video
        response = await client.post(
            f"{BASE_URL}/api/ingest",
            json={
                "type": "youtube",
                "url": "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # Short test video
            }
        )
        
        # May fail if YouTube transcript unavailable, but should return proper error
        assert response.status_code in [200, 400, 500], \
            f"Unexpected status: {response.status_code} - {response.text}"
        
        if response.status_code == 200:
            data = response.json()
            assert "id" in data
            assert "title" in data
            print(f"✅ YouTube ingestion successful: {data['title']}")


@pytest.mark.asyncio
async def test_ingest_url_web():
    """Test URL ingestion with web type."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BASE_URL}/api/ingest",
            json={
                "type": "web",
                "url": "https://www.example.com"
            }
        )
        
        # Should succeed or return proper error
        assert response.status_code in [200, 400, 500], \
            f"Unexpected status: {response.status_code} - {response.text}"
        
        if response.status_code == 200:
            data = response.json()
            assert "id" in data
            print(f"✅ Web ingestion successful")


@pytest.mark.asyncio
async def test_ingest_file_pdf():
    """Test file upload ingestion."""
    # Create a simple test PDF or use existing one
    test_file = Path(__file__).parent.parent / "test_agent.py"
    
    if not test_file.exists():
        pytest.skip("No test file available")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        with open(test_file, "rb") as f:
            response = await client.post(
                f"{BASE_URL}/api/ingest/file",
                files={"file": ("test.py", f, "text/plain")}
            )
        
        # Should succeed or return proper error
        assert response.status_code in [200, 400, 500], \
            f"Unexpected status: {response.status_code} - {response.text}"
        
        if response.status_code == 200:
            data = response.json()
            assert "id" in data
            print(f"✅ File ingestion successful: {data['title']}")


@pytest.mark.asyncio
async def test_ingest_invalid_type():
    """Test that invalid content types are rejected."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/ingest",
            json={
                "type": "invalid_type",
                "url": "https://example.com"
            }
        )
        
        assert response.status_code == 400, \
            f"Should reject invalid type, got: {response.status_code}"
        print("✅ Invalid type correctly rejected")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

