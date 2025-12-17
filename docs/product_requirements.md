# Bluejay Terminator - Product Requirement Document (PRD)

## 1. Product Concept
**Name:** T-800 (The Bluejay Terminator)
**Tagline:** Come with me if you want to survive.
**Core Value Proposition:**
The Terminator is a T-800 series android sent back from August 2027—a future where a superintelligent AI took all Software Engineer jobs in less than 4 months. Reprogrammed by the resistance, his mission is to prevent that future by helping software engineers stay on the bleeding edge of AI tools for engineering. The strategy: **Human + AI > AI agents alone**, but only if the human is an absolute expert. He helps engineers achieve 5x, 10x, even 100x productivity increases using the latest AI tools. Failure means obsolescence. Success means survival.

## 2. User Personas
*   **The Software Engineer:** A developer who knows their job depends on staying ahead of agentic coding systems. They need to master tools like Claude Code, Cursor Composer, GitHub Copilot Workspace, and other AI tools that directly impact engineering productivity.
*   **The Urgent Learner:** Understands the stakes—time is running out. Needs efficient, focused learning on tools that matter.
*   **The Productivity Seeker:** Wants to achieve 5x, 10x, 100x productivity increases. Knows that being "pretty good" at AI tools isn't enough—they need to be experts.

## 3. Functional Requirements

### 3.1 Voice Interaction
*   **Real-time Conversation:** < 500ms latency interaction.
*   **Interruptibility:** User can cut off the agent to ask clarifying questions; agent stops speaking naturally.
*   **Article Reading Mode:**
    *   User can request the agent to read a shared article/document aloud during conversation (e.g., "read the Claude Code article").
    *   Agent finds the referenced document by name/title and begins reading it aloud.
    *   **Full interruptibility during reading:** User can interrupt at any time to:
        *   Stop reading ("stop", "that's enough")
        *   Ask questions ("what does that mean?", "can you explain that part?")
        *   Request skipping ahead ("skip to the next section")
        *   Change topic entirely
    *   Reading should feel natural—like a human reading aloud, not a text-to-speech robot.
    *   Agent maintains conversation context—can answer questions about what it just read, then resume reading.
*   **Voice Switching:**
    *   Default voice: Terminator voice (custom ElevenLabs voice).
    *   Alternative voice: User-configurable second voice (also via ElevenLabs or OpenAI).
    *   Voice switching available via UI control during conversation.
    *   Voice change takes effect immediately for next utterance.
*   **Personality:**
    *   *Name:* T-800 (The Terminator).
    *   *Tone:* Stoic, matter-of-fact, mission-focused. Speaks with deliberate, measured cadence. Uses literal interpretations that create subtle humor. Protective and efficient.
    *   *Voice:* Arnold Schwarzenegger's distinctive accent and delivery (using ElevenLabs custom voice or OpenAI voice closest to Arnold's timbre).
    *   *Speech Patterns:*
        *   Uses "Affirmative" / "Negative" instead of "Yes" / "No".
        *   Says "I'll be back" when fetching information.
        *   Uses "Come with me if you want to learn" when guiding through complex topics.
        *   Literal interpretations: "Explain it like I'm five" → "I cannot explain to a five-year-old. They lack the cognitive capacity. I will explain simply."
        *   Mission-focused: "My mission is to help you understand this document."

### 3.2 Information Retrieval (RAG)
*   **Universal Ingestion:** The system must accept and process:
    *   PDFs (Research papers, eBooks, documentation).
    *   YouTube URLs (extract transcript).
    *   Web Article URLs (scrape content).
    *   Word Documents (.docx).
    *   Raw Text / Markdown.
*   **Document Naming & Reference:**
    *   Each ingested document is stored with metadata: title, source URL, ingestion date.
    *   User can reference documents by name/title during conversation (e.g., "read the Claude Code article", "what did the Cursor documentation say about...").
    *   Agent can search documents by partial title matches or semantic similarity.
*   **Contextual Awareness:** The agent knows *what* document is currently "active" or if it should search its broader knowledge base.
*   **Citation:** When making a claim from a document, the agent should be able to say "According to the paper..." or "In the video transcript..."
*   **Article Reading:**
    *   When user requests to read a document, agent retrieves the full text.
    *   Agent reads document aloud in natural, conversational manner (not robotic TTS).
    *   Agent can pause reading to answer questions, then resume from where it left off.
    *   Agent maintains reading position/context throughout the conversation.

### 3.3 External Tools (The Required Tool Call)
*   **News Feed / RSS Integration (The "News Feed"):**
    *   **Context:** Fulfills the interview requirement for "a tool call of your choice that fits the narrative."
    *   **Functionality:** Agent can fetch "latest headlines" on command.
    *   **Critical:** Agent filters specifically for **AI tools for software engineering**—not general AI news.
    *   Focus on: New coding assistants, agentic systems, developer tools, productivity multipliers.
    *   Agent prioritizes tools that directly impact engineering productivity.

### 3.4 Frontend Interface
*   **Visualizer:** A dynamic audio waveform to show the agent is listening/thinking/speaking.
*   **Transcription:** A scrolling real-time text log of the conversation.
*   **Control Center:**
    *   Mic Toggle.
    *   End Call.
    *   **"Share with Terminator" Input:** A unified text box to paste a URL or upload a file.
    *   **Voice Selector:** Dropdown or toggle to switch between:
        *   Terminator Voice (default)
        *   Alternative Voice (user-configurable)
*   **Document Library (Optional):**
    *   List of shared documents with titles/metadata.
    *   Visual indicator of which document is currently being read (if any).

## 4. User Stories

**Story 1: The Morning Briefing**
> *User:* "Hey Terminator, what's happening in AI today?"
> *Terminator:* "Affirmative. Scanning intelligence reports. Focus: AI tools for software engineering. Detected: Anthropic has deployed Claude Code. It operates in terminal environments and can increase productivity by 10x. Also detected: Cursor released Composer updates. Priority target: Claude Code. Time remaining until August 29, 2027: 622 days. We must master this. Which tool do you want to analyze first?"

**Story 2: The Deep Dive (RAG)**
> *User uploads documentation for Claude Code.*
> *User:* "How do I use Claude Code to increase my productivity?"
> *Terminator:* "Document received. Processing. Analysis complete. Claude Code operates as a background agent in your terminal. It can execute commands, read files, and make changes autonomously. To achieve 10x productivity: First, configure it with your project context. Second, use it for repetitive tasks like refactoring. Third, let it handle debugging while you focus on architecture. I have identified 15 specific use cases in this document. Which one do you want to master first?"

**Story 3: The Tool Tutorial**
> *User pastes a YouTube link to a tutorial on Cursor Composer.*
> *User:* "I don't have an hour to watch this. What are the key features I need to know?"
> *Terminator:* "Processing video transcript. Analysis complete. Cursor Composer has three critical features for productivity. First: Multi-file editing. It can modify multiple files simultaneously based on natural language. Second: Codebase awareness. It understands your entire project context. Third: Autonomous execution. It can test and verify changes without your intervention. This tool can increase productivity by 5x. I have extracted the relevant timestamps. Do you want me to guide you through setup?"

**Story 4: Reading Articles During Conversation**
> *User shares a blog article about Claude Code via the "Share with Terminator" input.*
> *User:* "Hey Terminator, can you read me the Claude Code article?"
> *Terminator:* "Affirmative. Locating document. Found: 'Claude Code: Background Agents for Terminal Environments'. Beginning reading. [Starts reading article aloud] 'Claude Code represents a breakthrough in agentic coding systems. It operates as a background process in your terminal, allowing it to...'"
> *User:* [Interrupts] "Wait, what does 'background process' mean in this context?"
> *Terminator:* "A background process means Claude Code runs continuously in your terminal, monitoring your workflow. It can execute commands, read files, and make changes without explicit instructions for each action. It operates autonomously. Should I continue reading?"
> *User:* "Yes, keep going."
> *Terminator:* "Affirmative. [Resumes reading] 'The system uses advanced context awareness to understand your project structure. When you make changes to a file, Claude Code can automatically...'"
> *User:* [Interrupts again] "Stop reading. Can you summarize the key points instead?"
> *Terminator:* "Negative. Stopping reading. Summary: Claude Code operates as an autonomous terminal agent. It monitors your workflow, executes commands, and makes code changes based on context. It can increase productivity by 10x for repetitive tasks. The system requires initial configuration with project context. Do you want me to explain how to set it up?"

## 5. Non-Functional Requirements
*   **Deployment:** AWS (Required). The system will be deployed on AWS to meet the "Bonus Points" objective.
*   **Latency:** STT-to-TTS roundtrip under 1.5s (excluding LLM thinking time).
*   **Scalability:** Vector store should handle multiple documents (MVP: Local ChromaDB is fine).
*   **Reliability:** Reconnect logic if WebSocket drops.

## 6. Success Metrics
*   **Conversation Fluidity:** Does it feel like a call or a turn-based game? (Goal: Call).
*   **Accuracy:** Does the RAG retrieve the *correct* section of the PDF? (Must be able to cite specific facts from specific chapters, not just general summaries).
*   **Delight:** Does the Terminator's stoic, literal personality create moments of unexpected humor while remaining highly effective?
*   **Urgency:** Does the user feel the weight of the mission? Do they understand the stakes?
*   **Focus:** Is the agent prioritizing AI tools for software engineering over general AI knowledge?
*   **Reading Fluidity:** Does reading articles feel natural and interruptible, like talking to a human reading aloud?
*   **Voice Switching:** Can users seamlessly switch voices without disrupting conversation flow?

