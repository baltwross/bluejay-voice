import { useEffect, useRef, useMemo } from 'react';
import { User, Bot, AlertTriangle, Loader2 } from 'lucide-react';
import { useVoiceAssistant, useTranscriptions, useLocalParticipant } from '@livekit/components-react';
import { cn, formatTime, generateId } from '../utils';
import type { TranscriptMessage } from '../types';

interface TranscriptProps {
  /** Custom class name for the container */
  className?: string;
  /** Whether there's an active connection */
  isConnected?: boolean;
}

interface TranscriptItemProps {
  message: TranscriptMessage;
}

/** Individual transcript message item */
const TranscriptItem = ({ message }: TranscriptItemProps) => {
  const isAgent = message.sender === 'agent';

  return (
    <div
      className={cn(
        'flex gap-3 p-3 rounded-lg transition-all duration-200',
        'border border-transparent',
        isAgent
          ? 'bg-terminator-red/5 hover:border-terminator-red/20'
          : 'bg-terminator-cyan/5 hover:border-terminator-cyan/20',
        !message.isFinal && 'opacity-70'
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center',
          isAgent
            ? 'bg-terminator-red/20 text-terminator-red'
            : 'bg-terminator-cyan/20 text-terminator-cyan'
        )}
      >
        {isAgent ? (
          <Bot className="w-4 h-4" />
        ) : (
          <User className="w-4 h-4" />
        )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        {/* Header */}
        <div className="flex items-center gap-2 mb-1">
          <span
            className={cn(
              'text-xs font-mono font-semibold tracking-wider',
              isAgent ? 'text-terminator-red' : 'text-terminator-cyan'
            )}
          >
            {isAgent ? 'T-800' : 'OPERATOR'}
          </span>
          <span className="text-[10px] font-mono text-terminator-text-dim">
            {formatTime(message.timestamp)}
          </span>
          {!message.isFinal && (
            <Loader2 className="w-3 h-3 text-terminator-text-dim animate-spin" />
          )}
        </div>

        {/* Message text */}
        <p
          className={cn(
            'text-sm font-mono leading-relaxed break-words',
            isAgent ? 'text-terminator-text' : 'text-terminator-text'
          )}
        >
          {message.text}
        </p>
      </div>
    </div>
  );
};

/** Empty state when no transcript */
const EmptyState = ({ isConnected }: { isConnected: boolean }) => (
  <div className="flex flex-col items-center justify-center h-full text-terminator-text-dim">
    <AlertTriangle className="w-12 h-12 mb-4 opacity-30" />
    <p className="font-mono text-sm">
      {isConnected ? 'AWAITING TRANSMISSION' : 'NO ACTIVE SESSION'}
    </p>
    <p className="text-xs mt-2 opacity-50">
      {isConnected
        ? 'Start speaking to initiate conversation'
        : 'Initialize connection to begin conversation'}
    </p>
  </div>
);

/**
 * Transcript - Displays real-time conversation transcript
 * 
 * Shows messages from both user and agent with auto-scrolling
 * and Terminator HUD styling.
 */
export const Transcript = ({ className, isConnected = false }: TranscriptProps) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const { agentTranscriptions } = useVoiceAssistant();
  const { localParticipant } = useLocalParticipant();
  const userTranscriptions = useTranscriptions({
    participantIdentities: localParticipant ? [localParticipant.identity] : [],
  });

  // Combine and sort transcriptions
  const messages = useMemo((): TranscriptMessage[] => {
    const combined: TranscriptMessage[] = [];

    // Process agent transcriptions (with safety check)
    if (agentTranscriptions && Array.isArray(agentTranscriptions)) {
      agentTranscriptions.forEach((segment) => {
        combined.push({
          id: segment.id || generateId(),
          sender: 'agent',
          text: segment.text,
          timestamp: new Date(segment.firstReceivedTime ?? Date.now()),
          isFinal: segment.final,
        });
      });
    }

    // Process user transcriptions (with safety check)
    if (userTranscriptions && Array.isArray(userTranscriptions)) {
      userTranscriptions.forEach((segment) => {
        combined.push({
          id: (segment as any).id || generateId(),
          sender: 'user',
          text: segment.text,
          timestamp: new Date((segment as any).firstReceivedTime || (segment as any).timestamp || Date.now()),
          isFinal: (segment as any).final ?? (segment as any).isFinal ?? true,
        });
      });
    }

    // #region agent log
    fetch('http://127.0.0.1:7243/ingest/e6f8272a-4cdc-4521-bede-fa9c0e8e594a', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        location: 'frontend/src/components/Transcript.tsx:messages',
        message: 'Transcript messages processed',
        data: { count: combined.length, timestamps: combined.map(m => m.timestamp.getTime()) },
        timestamp: Date.now(),
        sessionId: 'debug-session',
        hypothesisId: 'timestamp_jitter'
      })
    }).catch(()=>{});
    // #endregion

    // Sort by timestamp
    return combined.sort(
      (a, b) => a.timestamp.getTime() - b.timestamp.getTime()
    );
  }, [agentTranscriptions, userTranscriptions]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div
      className={cn(
        'hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-4 flex flex-col',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-4 pb-2 border-b border-terminator-border">
        <span className="text-terminator-red text-xs font-mono tracking-wider">
          ▸ TRANSCRIPT
        </span>
        <span className="flex-1" />
        {messages.length > 0 && (
          <span className="text-terminator-text-dim text-xs font-mono">
            {messages.length} ENTRIES
          </span>
        )}
      </div>

      {/* Messages List */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto space-y-3 min-h-0"
      >
        {messages.length > 0 ? (
          messages.map((message) => (
            <TranscriptItem key={message.id} message={message} />
          ))
        ) : (
          <EmptyState isConnected={isConnected} />
        )}
      </div>

      {/* Footer - Transcript stats */}
      <div className="pt-2 mt-2 border-t border-terminator-border">
        <div className="flex items-center justify-between text-[10px] font-mono text-terminator-text-dim">
          <span>
            USER: {userTranscriptions?.length ?? 0} | AGENT: {agentTranscriptions?.length ?? 0}
          </span>
          <span className={cn(
            'flex items-center gap-1',
            isConnected ? 'text-terminator-cyan' : 'text-terminator-text-dim'
          )}>
            <div className={cn(
              'w-1.5 h-1.5 rounded-full',
              isConnected ? 'bg-terminator-cyan animate-pulse' : 'bg-terminator-text-dim'
            )} />
            {isConnected ? 'LIVE' : 'OFFLINE'}
          </span>
        </div>
      </div>
    </div>
  );
};

