import { useCallback, useRef, useMemo } from 'react';
import { useVoiceAssistant, useTranscriptions, useLocalParticipant } from '@livekit/components-react';
import { generateId } from '../utils';
import type { TranscriptMessage } from '../types';

interface SaveTranscriptPayload {
  roomName: string;
  messages: Array<{
    id: string;
    sender: string;
    text: string;
    timestamp: string;
    isFinal: boolean;
  }>;
  startTime: string;
  endTime: string;
}

interface SaveTranscriptResponse {
  filename: string;
  messageCount: number;
  savedAt: string;
}

interface UseTranscriptOptions {
  /** Room name for the current session */
  roomName: string | null;
  /** API endpoint for saving transcripts */
  saveEndpoint?: string;
}

interface UseTranscriptReturn {
  /** Combined and sorted transcript messages */
  messages: TranscriptMessage[];
  /** Save the current transcript to the backend */
  saveTranscript: () => Promise<SaveTranscriptResponse | null>;
  /** Whether a save operation is in progress */
  isSaving: boolean;
}

/**
 * useTranscript - Hook for managing conversation transcripts
 * 
 * Combines user and agent transcriptions from LiveKit and provides
 * functionality to save transcripts to the backend for debugging.
 */
export function useTranscript(options: UseTranscriptOptions): UseTranscriptReturn {
  const { roomName, saveEndpoint = '/api/transcripts' } = options;
  
  const { agentTranscriptions } = useVoiceAssistant();
  const { localParticipant } = useLocalParticipant();
  const userTranscriptions = useTranscriptions({
    participantIdentities: localParticipant ? [localParticipant.identity] : [],
  });
  
  const isSavingRef = useRef(false);
  const sessionStartRef = useRef<Date>(new Date());

  // Combine and sort transcriptions
  const messages = useMemo((): TranscriptMessage[] => {
    const combined: TranscriptMessage[] = [];

    // Process agent transcriptions
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

    // Process user transcriptions
    if (userTranscriptions && Array.isArray(userTranscriptions)) {
      userTranscriptions.forEach((stream, index) => {
        const streamInfo = (stream as { streamInfo?: { timestamp?: number; id?: string } }).streamInfo;
        combined.push({
          id: streamInfo?.id || `user-${index}-${Date.now()}`,
          sender: 'user',
          text: stream.text,
          timestamp: new Date(streamInfo?.timestamp ?? Date.now()),
          isFinal: true,
        });
      });
    }

    // Sort by timestamp
    return combined.sort(
      (a, b) => a.timestamp.getTime() - b.timestamp.getTime()
    );
  }, [agentTranscriptions, userTranscriptions]);

  // Save transcript to backend
  const saveTranscript = useCallback(async (): Promise<SaveTranscriptResponse | null> => {
    // Don't save if no messages or already saving
    if (messages.length === 0 || isSavingRef.current) {
      console.log('Skipping transcript save: no messages or already saving');
      return null;
    }

    // Only save if we have a room name
    if (!roomName) {
      console.warn('Cannot save transcript: no room name available');
      return null;
    }

    isSavingRef.current = true;

    try {
      const payload: SaveTranscriptPayload = {
        roomName,
        messages: messages.map((msg) => ({
          id: msg.id,
          sender: msg.sender,
          text: msg.text,
          timestamp: msg.timestamp.toISOString(),
          isFinal: msg.isFinal,
        })),
        startTime: sessionStartRef.current.toISOString(),
        endTime: new Date().toISOString(),
      };

      const response = await fetch(saveEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const result: SaveTranscriptResponse = await response.json();
      console.log(`Transcript saved: ${result.filename} (${result.messageCount} messages)`);
      return result;
    } catch (error) {
      console.error('Failed to save transcript:', error);
      return null;
    } finally {
      isSavingRef.current = false;
    }
  }, [messages, roomName, saveEndpoint]);

  return {
    messages,
    saveTranscript,
    isSaving: isSavingRef.current,
  };
}

