/** Transcript message from user or agent */
export interface TranscriptMessage {
  id: string;
  sender: 'user' | 'agent';
  text: string;
  timestamp: Date;
  isFinal: boolean;
}

/** Connection state for the LiveKit room */
export type ConnectionState = 
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'error';

/** Agent state during conversation */
export type AgentState = 
  | 'idle'
  | 'listening'
  | 'thinking'
  | 'speaking'
  | 'reading';

/** Document metadata for RAG system */
export interface DocumentMetadata {
  id: string;
  title: string;
  sourceType: 'pdf' | 'youtube' | 'web' | 'text';
  sourceUrl?: string;
  ingestedAt: Date;
}

/** Voice configuration options */
export interface VoiceConfig {
  voiceId: 'terminator' | 'standard';
  label: string;
}

/** Available voices */
export const VOICE_OPTIONS: VoiceConfig[] = [
  { voiceId: 'terminator', label: 'T-800 (Terminator)' },
  { voiceId: 'standard', label: 'Standard Voice' },
];




