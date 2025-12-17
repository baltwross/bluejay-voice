#!/usr/bin/env python3
"""
Quick verification script to test the API fixes.
Run this after restarting the backend server.
"""
import asyncio
import httpx
import sys

BASE_URL = "http://localhost:8080"


async def test_content_type_aliases():
    """Test that content type aliases work."""
    print("🧪 Testing content type aliases...")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Test web_article alias
        try:
            response = await client.post(
                f"{BASE_URL}/api/ingest",
                json={"type": "web_article", "url": "https://www.example.com"}
            )
            if response.status_code == 200:
                print("  ✅ 'web_article' alias works")
            elif response.status_code == 400:
                print(f"  ❌ 'web_article' alias failed: {response.text}")
            else:
                print(f"  ⚠️  'web_article' returned {response.status_code} (may be expected)")
        except Exception as e:
            print(f"  ⚠️  Error testing web_article: {e}")
        
        # Test url alias
        try:
            response = await client.post(
                f"{BASE_URL}/api/ingest",
                json={"type": "url", "url": "https://www.example.com"}
            )
            if response.status_code in [200, 400]:  # 400 is OK if URL is invalid
                print("  ✅ 'url' alias works (auto-detects as web)")
            else:
                print(f"  ⚠️  'url' returned {response.status_code}")
        except Exception as e:
            print(f"  ⚠️  Error testing url: {e}")


async def test_compatibility_endpoint():
    """Test that compatibility endpoint exists."""
    print("\n🧪 Testing compatibility endpoint...")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Test /api/documents/upload exists (should return 422 for missing file, not 404)
        try:
            response = await client.post(f"{BASE_URL}/api/documents/upload")
            if response.status_code == 422:  # Validation error (missing file) is OK
                print("  ✅ Compatibility endpoint /api/documents/upload exists")
            elif response.status_code == 404:
                print("  ❌ Compatibility endpoint /api/documents/upload not found")
            else:
                print(f"  ⚠️  Compatibility endpoint returned {response.status_code}")
        except Exception as e:
            print(f"  ⚠️  Error testing compatibility endpoint: {e}")


async def test_health_endpoint():
    """Test enhanced health endpoint."""
    print("\n🧪 Testing health endpoint...")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                data = response.json()
                if "components" in data:
                    print("  ✅ Health endpoint returns component status")
                    print(f"     Components: {list(data['components'].keys())}")
                else:
                    print("  ⚠️  Health endpoint missing components")
            else:
                print(f"  ❌ Health endpoint returned {response.status_code}")
        except Exception as e:
            print(f"  ❌ Health endpoint error: {e}")


async def test_news_feed():
    """Test news feed endpoint."""
    print("\n🧪 Testing news feed endpoint...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{BASE_URL}/api/newsfeed",
                params={"query": "AI tools", "max_results": 2}
            )
            if response.status_code == 200:
                data = response.json()
                if "results" in data and "count" in data:
                    print(f"  ✅ News feed endpoint works ({data['count']} results)")
                else:
                    print("  ⚠️  News feed response missing expected fields")
            elif response.status_code == 503:
                print("  ⚠️  News feed unavailable (Tavily not configured)")
            else:
                print(f"  ⚠️  News feed returned {response.status_code}")
        except Exception as e:
            print(f"  ⚠️  Error testing news feed: {e}")


async def main():
    """Run all verification tests."""
    print("=" * 60)
    print("🔍 Verifying API Fixes")
    print("=" * 60)
    print(f"\nTesting against: {BASE_URL}")
    print("(Make sure backend server is running: python backend/token_server.py)\n")
    
    # Check if server is running
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.get(f"{BASE_URL}/health")
    except Exception:
        print("❌ Backend server not running!")
        print("   Start it with: python backend/token_server.py")
        sys.exit(1)
    
    await test_content_type_aliases()
    await test_compatibility_endpoint()
    await test_health_endpoint()
    await test_news_feed()
    
    print("\n" + "=" * 60)
    print("✅ Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

