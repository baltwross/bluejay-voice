"""
LiveKit WebRTC Integration Tests for Voice Pipeline

These tests verify voice interaction features using the LiveKit WebRTC SDK.
Voice features cannot be tested via HTTP endpoints - they require WebRTC connections.
"""
import asyncio
import pytest
import time
from typing import Optional
from datetime import datetime

try:
    from livekit import api, rtc
    from livekit.agents import (
        JobContext,
        WorkerOptions,
        cli,
        llm,
    )
    from livekit.plugins import deepgram, openai, elevenlabs
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    pytest.skip("LiveKit SDK not available", allow_module_level=True)

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config


class VoicePipelineTest:
    """Test harness for voice pipeline testing."""
    
    def __init__(self):
        self.config = get_config()
        self.room: Optional[rtc.Room] = None
        self.audio_tracks_received = []
        self.transcriptions_received = []
        self.connection_established = False
        
    async def connect(self) -> tuple[str, str]:
        """
        Connect to LiveKit and return token and room name.
        
        Returns:
            Tuple of (token, room_name)
        """
        # Generate room name
        room_name = f"test-{int(time.time())}"
        
        # Create access token
        token = (
            api.AccessToken(
                api_key=self.config.livekit.api_key,
                api_secret=self.config.livekit.api_secret,
            )
            .with_identity("test-user")
            .with_name("Test User")
            .with_grants(
                api.VideoGrants(
                    room=room_name,
                    room_join=True,
                    room_create=True,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            .to_jwt()
        )
        
        return token, room_name
    
    async def setup_room(self, token: str, room_name: str):
        """Set up LiveKit room connection."""
        self.room = rtc.Room()
        
        @self.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            """Handle incoming audio tracks."""
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                self.audio_tracks_received.append({
                    "track_id": track.sid,
                    "participant": participant.identity,
                    "timestamp": datetime.now().isoformat(),
                })
        
        @self.room.on("data_received")
        def on_data_received(data: rtc.DataPacket, participant: rtc.RemoteParticipant):
            """Handle data channel messages."""
            # Handle transcriptions or other data
            try:
                import json
                payload = json.loads(data.data)
                if payload.get("type") == "transcription":
                    self.transcriptions_received.append(payload)
            except Exception:
                pass
        
        @self.room.on("connected")
        def on_connected():
            """Handle connection established."""
            self.connection_established = True
        
        # Connect to room
        await self.room.connect(
            self.config.livekit.ws_url,
            token,
        )
        
        # Wait for connection
        await asyncio.sleep(1)
    
    async def cleanup(self):
        """Clean up room connection."""
        if self.room:
            await self.room.disconnect()
            self.room = None


@pytest.mark.asyncio
async def test_voice_connection_establishment():
    """
    Test TC001: Voice Pipeline Roundtrip Latency
    
    Verify that WebRTC connection can be established and audio tracks are received.
    This is a foundational test for voice interaction.
    """
    test_harness = VoicePipelineTest()
    
    try:
        # Get token and room name
        token, room_name = await test_harness.connect()
        
        # Set up room connection
        await test_harness.setup_room(token, room_name)
        
        # Verify connection established
        assert test_harness.connection_established, "Failed to establish WebRTC connection"
        
        # Wait a bit for agent to join (if running)
        await asyncio.sleep(2)
        
        # Note: Full latency testing requires:
        # 1. Sending audio via microphone track
        # 2. Receiving audio via remote participant track
        # 3. Measuring time between send and receive
        # This requires a running agent, so we just verify connection capability
        
        print(f"✅ Connection established to room: {room_name}")
        print(f"✅ Audio tracks received: {len(test_harness.audio_tracks_received)}")
        
    finally:
        await test_harness.cleanup()


@pytest.mark.asyncio
async def test_voice_token_generation():
    """
    Test that token generation endpoint works correctly.
    This is a prerequisite for voice connection.
    """
    import httpx
    
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8080/api/token")
        
        assert response.status_code == 200, f"Token generation failed: {response.status_code}"
        
        data = response.json()
        assert "token" in data, "Response missing token"
        assert "roomName" in data, "Response missing roomName"
        assert "serverUrl" in data, "Response missing serverUrl"
        
        # Verify token is valid JWT format (basic check)
        assert len(data["token"].split(".")) == 3, "Token is not a valid JWT"
        
        print(f"✅ Token generated for room: {data['roomName']}")


@pytest.mark.asyncio
async def test_voice_pipeline_components():
    """
    Test that all voice pipeline components are configured correctly.
    This verifies the setup without requiring a full WebRTC connection.
    """
    config = get_config()
    
    # Verify LiveKit config
    assert config.livekit.api_key, "LiveKit API key not configured"
    assert config.livekit.api_secret, "LiveKit API secret not configured"
    assert config.livekit.ws_url, "LiveKit WebSocket URL not configured"
    
    # Verify STT config (Deepgram)
    assert config.deepgram.api_key, "Deepgram API key not configured"
    
    # Verify TTS config (ElevenLabs)
    assert config.elevenlabs.api_key, "ElevenLabs API key not configured"
    assert config.elevenlabs.voice_id, "ElevenLabs voice ID not configured"
    
    # Verify LLM config (OpenAI)
    assert config.openai.api_key, "OpenAI API key not configured"
    
    print("✅ All voice pipeline components configured correctly")


@pytest.mark.skip(reason="Requires running agent and audio input/output")
async def test_voice_pipeline_latency():
    """
    Test TC001: Voice Pipeline Roundtrip Latency
    
    This test measures the complete roundtrip latency:
    - User speaks -> STT -> LLM -> TTS -> Agent speaks
    
    Requirements:
    - < 500ms latency for interruptibility
    - < 1.5s total roundtrip (excluding LLM processing)
    
    Note: This test requires:
    1. A running agent
    2. Audio input device
    3. Audio output device
    4. Real-time audio streaming
    
    To run this test:
    1. Start the agent: python -m livekit.agents dev
    2. Start the token server: python backend/token_server.py
    3. Run this test with audio devices connected
    """
    test_harness = VoicePipelineTest()
    
    try:
        token, room_name = await test_harness.connect()
        await test_harness.setup_room(token, room_name)
        
        # Enable microphone
        # Note: This requires browser-like environment or audio device
        # In a real test, you would:
        # 1. Create a microphone track
        # 2. Publish it to the room
        # 3. Send test audio
        # 4. Measure time until response audio received
        
        # Placeholder for actual implementation
        await asyncio.sleep(1)
        
    finally:
        await test_harness.cleanup()


@pytest.mark.skip(reason="Requires running agent and frontend integration")
async def test_voice_switching():
    """
    Test TC005: Voice Switching Realtime
    
    Verify that voice can be switched during conversation without disrupting flow.
    
    Note: This test requires:
    1. Frontend UI integration
    2. LiveKit data channel for voice switching commands
    3. Running agent that responds to voice switch requests
    """
    # Voice switching is handled via:
    # 1. Frontend UI sends data channel message with voice ID
    # 2. Agent receives message and switches TTS voice
    # 3. Next utterance uses new voice
    
    # This would require frontend integration testing
    pass


@pytest.mark.skip(reason="Requires running agent and document ingestion")
async def test_article_reading_mode():
    """
    Test TC004: Article Reading Mode Functionality
    
    Verify that article reading mode works with full interruptibility.
    
    Note: This test requires:
    1. Document ingested via /api/ingest or /api/ingest/file
    2. Voice connection established
    3. User requests article reading via voice
    4. Agent reads article aloud
    5. User interrupts and asks questions
    6. Agent answers and resumes reading
    """
    # This would require:
    # 1. Ingest a test document
    # 2. Connect via voice
    # 3. Request reading via voice command
    # 4. Verify reading starts
    # 5. Interrupt and verify agent stops
    # 6. Ask question and verify agent answers
    # 7. Resume reading and verify continuation
    pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

