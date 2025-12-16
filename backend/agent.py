import logging
import os
from dotenv import load_dotenv

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, elevenlabs, silero
from livekit.rtc import DataPacket, DataPacketKind, Participant

from prompts import SYSTEM_PROMPT
from rag.retriever import query_knowledge_base, find_relevant_document_content
from rag.indexer import index_document
from tools.news import get_latest_ai_news

import json
import asyncio

load_dotenv()

logger = logging.getLogger("bluejay-terminator")

def prewarm(proc: JobProcess):
    proc.userdata["preload_data"] = "..."

async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=SYSTEM_PROMPT,
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # Configure Tools
    # We will wrap the RAG function in an llm.FunctionContext
    
    class TerminatorTools(llm.FunctionContext):
        def __init__(self):
            super().__init__()
            self.agent = None

        @llm.ai_callable(description="Query the internal knowledge base / RAG for specific information from uploaded documents.")
        def query_documents(self, query: str):
            """
            Called when the user asks a specific question that might be answered by uploaded documents.
            """
            logger.info(f"Querying documents for: {query}")
            return query_knowledge_base(query)

        @llm.ai_callable(description="Fetch latest headlines about AI tools for software engineering.")
        def get_news(self):
            """
            Called when the user asks for news, updates, or "what's happening in AI".
            """
            logger.info("Fetching news...")
            return get_latest_ai_news()
        
        @llm.ai_callable(description="Read a document or article aloud to the user.")
        async def read_article(self, article_name_or_topic: str):
            """
            Called when the user asks to read an article or document.
            """
            logger.info(f"Request to read article: {article_name_or_topic}")
            content = find_relevant_document_content(article_name_or_topic)
            if content:
                if self.agent:
                    import asyncio
                    # We use a task to not block the LLM response
                    # We split content into chunks to ensure smoother interruptibility and processing
                    # But agent.say handles string well. Let's just pass it for now.
                    # To ensure "Reading mode", we might want to say "Reading..." first.
                    asyncio.create_task(self.agent.say(f"Reading {article_name_or_topic}: {content}", allow_interruptions=True))
                    return "Affirmative. Beginning playback."
                return "Error. Agent reference not found."
            return "Negative. Document not found."

    tools = TerminatorTools()
    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=elevenlabs.TTS(),
        chat_ctx=initial_ctx,
        fnc_ctx=tools,
    )
    tools.agent = agent

    agent.start(ctx.room, participant)

    @ctx.room.on("data_received")
    def on_data_received(data: DataPacket, participant: Participant, kind: DataPacketKind):
        try:
             text = data.data.decode("utf-8")
             payload = json.loads(text)
             if payload.get("type") == "USER_INPUT":
                 content = payload.get("content")
                 # Check if URL
                 if content.startswith("http"):
                     # Trigger ingestion
                     logger.info(f"Ingesting URL: {content}")
                     # Ingestion might be blocking/slow, run in thread or if index_document is sync?
                     # index_document is sync. We should run it in executor or accept it blocks slightly.
                     # Better: use asyncio.to_thread
                     
                     async def handle_ingest():
                        await asyncio.to_thread(index_document, content)
                        await agent.say(f"Affirmative. Intelligence received. I have analyzed {content}.", allow_interruptions=True)
                     
                     asyncio.create_task(handle_ingest())
                 else:
                     logger.info(f"Received text message: {content}")
                     # Optional: Handle text chat
        except Exception as e:
            logger.error(f"Error handling data: {e}")

    await agent.say("I am the Terminator. I have been sent back to prevent your obsolescence. State your mission.", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

