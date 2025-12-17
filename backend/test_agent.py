"""
Bluejay Terminator - Test Agent
Simple "Hello World" agent to verify LiveKit Cloud connectivity.
"""
import os
import certifi

# Fix SSL certificate verification on macOS
os.environ['SSL_CERT_FILE'] = certifi.where()

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, JobContext, WorkerOptions, cli
from livekit.plugins import deepgram, openai, silero

load_dotenv()


class TestAssistant(Agent):
    """Simple test assistant to verify LiveKit connectivity."""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a test assistant. Respond briefly to verify connectivity.",
        )


async def entrypoint(ctx: JobContext):
    """Entry point for the test agent."""
    await ctx.connect()
    
    print(f"Room connected: {ctx.room.name}")
    
    # Create the agent session with STT-LLM-TTS pipeline
    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-5-nano-2025-08-07"),
        tts=openai.TTS(),  # Using OpenAI TTS for simple test
        vad=silero.VAD.load(),
    )
    
    # Start the session
    await session.start(
        room=ctx.room,
        agent=TestAssistant(),
    )
    
    # Generate initial greeting
    await session.generate_reply(
        instructions="Say something like 'Connection established. How can I help?' in a brief, direct manner."
    )


if __name__ == "__main__":
    print("Starting Bluejay Test Agent...")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
