import { useState, useCallback, useEffect, useRef } from 'react';
import { Skull, AlertTriangle, Wifi, WifiOff } from 'lucide-react';
import {
  LiveKitRoom,
  RoomAudioRenderer,
  useConnectionState,
  useChat,
} from '@livekit/components-react';
import '@livekit/components-styles';

import { cn } from './utils';
import { useConnection, useDocuments, useTranscript } from './hooks';
import {
  AgentVisualizer,
  Transcript,
  ControlPanel,
  StandaloneControlPanel,
  InputConsole,
} from './components';
import type { ConnectionState } from './types';

// Get backend URL from env (for production) or default to empty string (relative path for dev proxy)
const BACKEND_URL = import.meta.env.VITE_API_URL || '';
const API_BASE = `${BACKEND_URL}/api`;

/**
 * Main Application Shell
 * Handles initial connection state before LiveKitRoom is mounted
 */
export const App = () => {
  const {
    connectionState: tokenState,
    token,
    roomName,
    serverUrl,
    connect,
    disconnect: disconnectToken,
    error: tokenError,
  } = useConnection({
    tokenEndpoint: `${API_BASE}/token`,
    autoReconnect: false,
  });

  const [shouldConnect, setShouldConnect] = useState(false);

  // Handle connect request
  const handleConnect = useCallback(async () => {
    await connect();
    setShouldConnect(true);
  }, [connect]);

  // Handle disconnect request
  const handleDisconnect = useCallback(() => {
    setShouldConnect(false);
    disconnectToken();
  }, [disconnectToken]);

  // Auto-connect on page load (runs once on mount)
  const hasAutoConnected = useRef(false);
  useEffect(() => {
    if (!hasAutoConnected.current) {
      hasAutoConnected.current = true;
      handleConnect();
    }
  }, [handleConnect]);

  // Determine effective connection state
  const effectiveConnectionState: ConnectionState =
    tokenState === 'connected' && shouldConnect && token
      ? 'connected'
      : tokenState;

  return (
    <div className="relative min-h-screen bg-terminator-darker overflow-hidden">
      {/* Scanline Overlay */}
      <div className="scanline-overlay" />

      {/* Background Grid */}
      <div
        className="absolute inset-0 opacity-5"
        style={{
          backgroundImage: `
            linear-gradient(rgba(255, 0, 51, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 0, 51, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px',
        }}
      />

      {/* Main Content */}
      <div className="relative z-10 flex flex-col h-screen p-2 sm:p-4 md:p-6">
        {/* Header */}
        <Header connectionState={effectiveConnectionState} />

        {/* Error Banner */}
        {tokenError && (
          <div className="mb-4 p-3 bg-terminator-red/10 border border-terminator-red/30 rounded-lg flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-terminator-red" />
            <span className="text-sm font-mono text-terminator-red">{tokenError}</span>
          </div>
        )}

        {/* Main Content Area */}
        {shouldConnect && token && serverUrl ? (
          <LiveKitRoom
            token={token}
            serverUrl={serverUrl}
            connect={true}
            audio={true}
            video={false}
            onDisconnected={handleDisconnect}
            className="flex-1 flex flex-col min-h-0"
          >
            <RoomAudioRenderer />
            <ConnectedContent onDisconnect={handleDisconnect} roomName={roomName} />
          </LiveKitRoom>
        ) : (
          <DisconnectedContent
            connectionState={effectiveConnectionState}
            onConnect={handleConnect}
            onDisconnect={handleDisconnect}
          />
        )}

        {/* Footer */}
        <Footer />
      </div>
    </div>
  );
};

/**
 * Header Component
 */
interface HeaderProps {
  connectionState: ConnectionState;
}

const Header = ({ connectionState }: HeaderProps) => {
  const isConnected = connectionState === 'connected';

  return (
    <header className="flex items-center justify-between mb-3 sm:mb-4 md:mb-6">
      <div className="flex items-center gap-2 sm:gap-3">
        <Skull className="w-6 h-6 sm:w-8 sm:h-8 text-terminator-red animate-pulse" />
        <div>
          <h1 className="font-display text-lg sm:text-xl md:text-2xl font-bold text-terminator-red text-glow-red tracking-wider">
            T-800
          </h1>
          <p className="text-[10px] sm:text-xs text-terminator-text-dim font-mono tracking-widest hidden xs:block">
            CYBERDYNE SYSTEMS MODEL 101
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 sm:gap-4">
        {/* Connection Status */}
        <div
          className={cn(
            'flex items-center gap-1.5 sm:gap-2 px-2 sm:px-3 py-1 sm:py-1.5 rounded border text-[10px] sm:text-xs',
            isConnected
              ? 'border-terminator-cyan/50 text-terminator-cyan'
              : 'border-terminator-border text-terminator-text-dim'
          )}
        >
          {isConnected ? (
            <Wifi className="w-3 h-3 sm:w-4 sm:h-4 animate-pulse" />
          ) : (
            <WifiOff className="w-3 h-3 sm:w-4 sm:h-4" />
          )}
          <span className="uppercase tracking-wider font-mono">
            {connectionState}
          </span>
        </div>
      </div>
    </header>
  );
};

/**
 * Connected Content - Rendered inside LiveKitRoom
 */
interface ConnectedContentProps {
  onDisconnect: () => void;
  roomName: string | null;
}

const ConnectedContent = ({ onDisconnect, roomName }: ConnectedContentProps) => {
  const roomConnectionState = useConnectionState();
  
  // Get backend URL for documents hook
  const BACKEND_URL = import.meta.env.VITE_API_URL || '';
  const API_BASE = `${BACKEND_URL}/api`;
  
  const { ingest, isIngesting, documents } = useDocuments({
    ingestEndpoint: `${API_BASE}/ingest`,
    listEndpoint: `${API_BASE}/documents`,
  });
  const { send: sendChatMessage } = useChat();
  const { messages: transcriptMessages, saveTranscript } = useTranscript({ 
    roomName,
    saveEndpoint: `${API_BASE}/transcripts`,
  });

  // Map LiveKit connection state to our type
  const connectionState: ConnectionState =
    roomConnectionState === 'connected'
      ? 'connected'
      : roomConnectionState === 'connecting'
      ? 'connecting'
      : roomConnectionState === 'reconnecting'
      ? 'reconnecting'
      : 'disconnected';

  // Handle disconnect with transcript saving
  const handleDisconnectWithSave = useCallback(async () => {
    // Save transcript before disconnecting
    if (transcriptMessages.length > 0) {
      console.log(`Saving transcript with ${transcriptMessages.length} messages...`);
      await saveTranscript();
    }
    onDisconnect();
  }, [transcriptMessages.length, saveTranscript, onDisconnect]);

  // Handle content ingestion and notify the agent
  const handleIngest = useCallback(
    async (input: { type: 'pdf' | 'youtube' | 'web' | 'file'; value: string | File }) => {
      try {
        const result = await ingest(input);
        
        // Notify the agent about the newly ingested document via chat
        const typeLabel = input.type === 'youtube' ? 'YouTube video' 
          : input.type === 'pdf' ? 'PDF document'
          : input.type === 'web' ? 'web article'
          : 'file';
        
        const notificationMessage = `[SYSTEM] I just shared a ${typeLabel} with you: "${result.title}". It has been added to your knowledge base.`;
        
        try {
          await sendChatMessage(notificationMessage);
          console.log('Agent notified of new document:', result.title);
        } catch (chatError) {
          console.warn('Failed to notify agent via chat:', chatError);
          // Don't fail the ingestion if chat notification fails
        }
      } catch (error) {
        console.error('Ingestion failed:', error);
        throw error;
      }
    },
    [ingest, sendChatMessage]
  );

  return (
    <main className="flex-1 flex flex-col lg:grid lg:grid-cols-3 gap-2 sm:gap-3 lg:gap-4 min-h-0 overflow-hidden">
      {/* Mobile: Controls Column First (order-1), Desktop: Right Column (order-2) */}
      <div className="flex flex-col gap-2 sm:gap-3 order-1 lg:order-2 flex-shrink-0 lg:flex-shrink lg:min-h-0 lg:overflow-y-auto lg:pr-2">
        {/* Agent Visualizer - Compact on mobile */}
        <AgentVisualizer className="flex-shrink-0 min-h-[120px] h-[min(140px,20vh)] sm:min-h-[160px] sm:h-[min(180px,25vh)] lg:min-h-[200px] lg:h-[min(240px,30vh)]" />

        {/* Control Panel */}
        <ControlPanel
          connectionState={connectionState}
          onConnect={() => {}}
          onDisconnect={handleDisconnectWithSave}
          className="flex-shrink-0"
        />

        {/* Input Console - Collapsible or smaller on mobile */}
        <InputConsole
          onSubmit={handleIngest}
          disabled={isIngesting || connectionState !== 'connected'}
          documents={documents}
          className="flex-shrink-0 hidden sm:block lg:block"
        />
      </div>

      {/* Mobile: Transcript Below (order-2), Desktop: Left Column (order-1, 2 cols) */}
      <div className="lg:col-span-2 order-2 lg:order-1 min-h-0 flex flex-col flex-1">
        <Transcript className="flex-1 min-h-0" isConnected={connectionState === 'connected'} />
      </div>

      {/* Mobile-only: Compact Input Console at bottom */}
      <div className="order-3 sm:hidden flex-shrink-0">
        <InputConsole
          onSubmit={handleIngest}
          disabled={isIngesting || connectionState !== 'connected'}
          documents={documents}
          className="flex-shrink-0"
        />
      </div>
    </main>
  );
};

/**
 * Disconnected Content - Rendered when not connected
 */
interface DisconnectedContentProps {
  connectionState: ConnectionState;
  onConnect: () => void;
  onDisconnect: () => void;
}

const DisconnectedContent = ({
  connectionState,
  onConnect,
  onDisconnect,
}: DisconnectedContentProps) => {
  return (
    <main className="flex-1 flex flex-col lg:grid lg:grid-cols-3 gap-2 sm:gap-3 lg:gap-4 min-h-0 overflow-hidden">
      {/* Mobile: Controls Column First (order-1), Desktop: Right Column (order-2) */}
      <div className="flex flex-col gap-2 sm:gap-3 order-1 lg:order-2 flex-shrink-0 lg:flex-shrink lg:overflow-y-auto lg:min-h-0 lg:pr-2">
        {/* Placeholder Visualizer - compact on mobile */}
        <div className="hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-2 sm:p-3 flex-shrink-0 flex flex-col overflow-hidden min-h-[100px] h-[min(120px,18vh)] sm:min-h-[140px] sm:h-[min(160px,22vh)] lg:min-h-[180px] lg:h-[min(220px,28vh)]">
          {/* Header */}
          <div className="w-full flex items-center gap-2 mb-1 sm:mb-2 pb-1 sm:pb-2 border-b border-terminator-border flex-shrink-0">
            <span className="text-terminator-red text-[9px] sm:text-xs font-mono tracking-wider">
              ▸ T-800 NEURAL NET
            </span>
            <span className="flex-1" />
            <span className="text-[10px] sm:text-xs font-mono tracking-wider text-terminator-text-dim">
              IDLE
            </span>
          </div>
          {/* Content */}
          <div className="flex-1 flex flex-col items-center justify-center min-h-0">
            <div
              className={cn(
                'w-12 h-12 sm:w-16 sm:h-16 md:w-20 md:h-20 mx-auto rounded-full border-2 flex items-center justify-center',
                'border-terminator-border'
              )}
            >
              <Skull className="w-6 h-6 sm:w-8 sm:h-8 md:w-10 md:h-10 text-terminator-text-dim" />
            </div>
            <p className="mt-1 sm:mt-2 font-mono text-[9px] sm:text-[10px] text-terminator-text-dim tracking-wider">
              STANDBY
            </p>
          </div>
          {/* Footer */}
          <div className="w-full pt-1 mt-1 border-t border-terminator-border flex items-center justify-center flex-shrink-0">
            <div className="flex items-center gap-1.5 sm:gap-2">
              <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 rounded-full bg-terminator-text-dim" />
              <span className="text-[8px] sm:text-[10px] font-mono text-terminator-text-dim tracking-widest">
                AWAITING CONNECTION
              </span>
            </div>
          </div>
        </div>

        {/* Control Panel */}
        <StandaloneControlPanel
          connectionState={connectionState}
          onConnect={onConnect}
          onDisconnect={onDisconnect}
          className="flex-shrink-0"
        />

        {/* Disabled Input Console - hidden on mobile when disconnected */}
        <InputConsole onSubmit={async () => {}} disabled className="flex-shrink-0 hidden sm:block" />
      </div>

      {/* Mobile: Transcript Placeholder Below (order-2), Desktop: Left Column (order-1, 2 cols) */}
      <div className="lg:col-span-2 order-2 lg:order-1 min-h-0 flex flex-col flex-1">
        {/* Transcript Placeholder */}
        <div className="hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-2 sm:p-3 lg:p-4 flex flex-col flex-1 min-h-0">
          <div className="flex items-center gap-2 mb-2 sm:mb-3 lg:mb-4 pb-2 border-b border-terminator-border">
            <span className="text-terminator-red text-[10px] sm:text-xs font-mono tracking-wider">
              ▸ TRANSCRIPT
            </span>
          </div>

          <div className="flex-1 flex items-center justify-center min-h-0">
            <div className="text-center">
              <AlertTriangle className="w-8 h-8 sm:w-10 sm:h-10 lg:w-12 lg:h-12 mx-auto mb-2 sm:mb-3 lg:mb-4 text-terminator-text-dim opacity-30" />
              <p className="font-mono text-xs sm:text-sm text-terminator-text-dim">NO ACTIVE SESSION</p>
              <p className="text-[10px] sm:text-xs mt-1 sm:mt-2 text-terminator-text-dim opacity-50">
                Initialize connection to begin conversation
              </p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
};

/**
 * Footer Component
 */
const Footer = () => {
  return (
    <footer className="mt-2 sm:mt-3 lg:mt-4 pt-2 sm:pt-3 lg:pt-4 border-t border-terminator-border">
      <div className="flex items-center justify-between text-[9px] sm:text-[10px] lg:text-xs font-mono text-terminator-text-dim">
        <span className="hidden sm:inline">MISSION: PREVENT AUGUST 29, 2027</span>
        <span className="sm:hidden">MISSION ACTIVE</span>
        <span className="text-terminator-red animate-blink">●</span>
        <span>DAYS: {calculateDaysRemaining()}</span>
      </div>
    </footer>
  );
};

/** Calculate days until August 29, 2027 */
function calculateDaysRemaining(): number {
  const targetDate = new Date('2027-08-29');
  const today = new Date();
  const diffTime = targetDate.getTime() - today.getTime();
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  return Math.max(0, diffDays);
}
