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
    tokenEndpoint: '/api/token',
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
      <div className="relative z-10 flex flex-col h-screen p-4 md:p-6">
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
    <header className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-3">
        <Skull className="w-8 h-8 text-terminator-red animate-pulse" />
        <div>
          <h1 className="font-display text-xl md:text-2xl font-bold text-terminator-red text-glow-red tracking-wider">
            T-800
          </h1>
          <p className="text-xs text-terminator-text-dim font-mono tracking-widest">
            CYBERDYNE SYSTEMS MODEL 101
          </p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Connection Status */}
        <div
          className={cn(
            'flex items-center gap-2 px-3 py-1.5 rounded border',
            isConnected
              ? 'border-terminator-cyan/50 text-terminator-cyan'
              : 'border-terminator-border text-terminator-text-dim'
          )}
        >
          {isConnected ? (
            <Wifi className="w-4 h-4 animate-pulse" />
          ) : (
            <WifiOff className="w-4 h-4" />
          )}
          <span className="text-xs uppercase tracking-wider font-mono">
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
  const { ingest, isIngesting, documents } = useDocuments();
  const { send: sendChatMessage } = useChat();
  const { messages: transcriptMessages, saveTranscript } = useTranscript({ roomName });

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
    <main className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0">
      {/* Left Panel - Transcript */}
      <div className="lg:col-span-2 min-h-0">
        <Transcript className="h-full" isConnected={connectionState === 'connected'} />
      </div>

      {/* Right Panel - Controls & Visualizer */}
      <div className="flex flex-col gap-4 min-h-0 overflow-y-auto pr-2">
        {/* Agent Visualizer */}
        <AgentVisualizer className="flex-grow flex-shrink-0 min-h-[300px]" />

        {/* Control Panel */}
        <ControlPanel
          connectionState={connectionState}
          onConnect={() => {}}
          onDisconnect={handleDisconnectWithSave}
          className="flex-shrink-0"
        />

        {/* Input Console */}
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
    <main className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0">
      {/* Left Panel - Placeholder */}
      <div className="lg:col-span-2 hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-4 flex flex-col">
        <div className="flex items-center gap-2 mb-4 pb-2 border-b border-terminator-border">
          <span className="text-terminator-red text-xs font-mono tracking-wider">
            ▸ TRANSCRIPT
          </span>
        </div>

        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <AlertTriangle className="w-12 h-12 mx-auto mb-4 text-terminator-text-dim opacity-30" />
            <p className="font-mono text-sm text-terminator-text-dim">NO ACTIVE SESSION</p>
            <p className="text-xs mt-2 text-terminator-text-dim opacity-50">
              Initialize connection to begin conversation
            </p>
          </div>
        </div>
      </div>

      {/* Right Panel */}
      <div className="flex flex-col gap-4 overflow-y-auto min-h-0 pr-2">
        {/* Placeholder Visualizer */}
        <div className="hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-4 flex-grow flex-shrink-0 flex items-center justify-center min-h-[300px]">
          <div className="text-center">
            <div
              className={cn(
                'w-24 h-24 md:w-32 md:h-32 mx-auto rounded-full border-2 flex items-center justify-center',
                'border-terminator-border'
              )}
            >
              <Skull className="w-12 h-12 md:w-16 md:h-16 text-terminator-text-dim" />
            </div>
            <p className="mt-4 font-display text-sm text-terminator-text-dim tracking-wider">
              STANDBY
            </p>
          </div>
        </div>

        {/* Control Panel */}
        <StandaloneControlPanel
          connectionState={connectionState}
          onConnect={onConnect}
          onDisconnect={onDisconnect}
          className="flex-shrink-0"
        />

        {/* Disabled Input Console */}
        <InputConsole onSubmit={async () => {}} disabled className="flex-shrink-0" />
      </div>
    </main>
  );
};

/**
 * Footer Component
 */
const Footer = () => {
  return (
    <footer className="mt-4 pt-4 border-t border-terminator-border">
      <div className="flex items-center justify-between text-xs font-mono text-terminator-text-dim">
        <span>MISSION: PREVENT AUGUST 29, 2027</span>
        <span className="text-terminator-red animate-blink">●</span>
        <span>DAYS REMAINING: {calculateDaysRemaining()}</span>
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
