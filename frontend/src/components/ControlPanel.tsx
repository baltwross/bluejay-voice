import { useCallback, useEffect, useState } from 'react';
import {
  Mic,
  MicOff,
  PhoneOff,
  Phone,
  Loader2,
} from 'lucide-react';
import { useLocalParticipant } from '@livekit/components-react';
import { cn } from '../utils';
import type { ConnectionState } from '../types';

interface ControlPanelProps {
  /** Current connection state */
  connectionState: ConnectionState;
  /** Callback when user wants to connect */
  onConnect: () => void;
  /** Callback when user wants to disconnect */
  onDisconnect: () => void;
  /** Custom class name */
  className?: string;
}

/**
 * ControlPanel - Provides call controls for the voice agent
 * 
 * Includes:
 * - Start/End call buttons
 * - Microphone mute/unmute toggle
 * - Connection status indicators
 */
export const ControlPanel = ({
  connectionState,
  onConnect,
  onDisconnect,
  className,
}: ControlPanelProps) => {
  const isConnected = connectionState === 'connected';
  const isConnecting = connectionState === 'connecting' || connectionState === 'reconnecting';
  const canConnect = connectionState === 'disconnected' || connectionState === 'error';

  return (
    <div
      className={cn(
        'hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-4',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-4 pb-2 border-b border-terminator-border">
        <span className="text-terminator-red text-xs font-mono tracking-wider">
          ▸ CONTROLS
        </span>
        <span className="flex-1" />
        <span
          className={cn(
            'text-xs font-mono tracking-wider px-2 py-0.5 rounded',
            isConnected && 'bg-terminator-cyan/20 text-terminator-cyan',
            isConnecting && 'bg-yellow-500/20 text-yellow-400',
            !isConnected && !isConnecting && 'bg-terminator-border text-terminator-text-dim'
          )}
        >
          {connectionState.toUpperCase()}
        </span>
      </div>

      {/* Main Controls */}
      <div className="space-y-3">
        {/* Primary Action Button */}
        {canConnect ? (
          <button
            onClick={onConnect}
            className={cn(
              'w-full btn-hud-primary flex items-center justify-center gap-2',
              'group relative overflow-hidden'
            )}
          >
            <Phone className="w-4 h-4 transition-transform group-hover:scale-110" />
            <span>Initialize Connection</span>
            {/* Animated border effect */}
            <span className="absolute inset-0 border border-terminator-red opacity-0 group-hover:opacity-100 transition-opacity animate-pulse" />
          </button>
        ) : isConnecting ? (
          <button
            disabled
            className="w-full btn-hud flex items-center justify-center gap-2 cursor-wait"
          >
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Establishing Link...</span>
          </button>
        ) : null}

        {/* Connected Controls */}
        {isConnected && (
          <div className="grid grid-cols-2 gap-2">
            {/* Mic Toggle */}
            <MicToggleButton />

            {/* End Call */}
            <button
              onClick={onDisconnect}
              className="btn-hud-danger flex items-center justify-center gap-2"
            >
              <PhoneOff className="w-4 h-4" />
              <span>End Call</span>
            </button>
          </div>
        )}
      </div>

      {/* Keyboard Shortcuts Info */}
      {isConnected && (
        <div className="mt-4 pt-3 border-t border-terminator-border">
          <p className="text-[10px] font-mono text-terminator-text-dim text-center">
            Press <kbd className="px-1.5 py-0.5 bg-terminator-border rounded text-[9px]">M</kbd> to toggle mic
            {' • '}
            <kbd className="px-1.5 py-0.5 bg-terminator-border rounded text-[9px]">ESC</kbd> to end call
          </p>
        </div>
      )}
    </div>
  );
};

/**
 * Mic toggle button that uses LiveKit's local participant
 */
const MicToggleButton = () => {
  const { localParticipant, isMicrophoneEnabled } = useLocalParticipant();
  const [isPending, setIsPending] = useState(false);

  const toggleMic = useCallback(async () => {
    if (!localParticipant) {
      return;
    }
    
    setIsPending(true);
    try {
      const targetState = !isMicrophoneEnabled;
      
      // If enabling mic, request permissions first
      if (targetState && navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          // Stop the stream immediately - we just needed permission
          stream.getTracks().forEach(track => track.stop());
        } catch (permError) {
          console.error('Microphone permission denied:', permError);
          alert('Microphone permission is required. Please allow microphone access in your browser settings.');
          return;
        }
      }
      
      await localParticipant.setMicrophoneEnabled(targetState);
    } catch (error) {
      console.error('Failed to toggle microphone:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      if (errorMessage.includes('Permission') || errorMessage.includes('permission')) {
        alert('Microphone permission is required. Please allow microphone access in your browser settings.');
      }
    } finally {
      setIsPending(false);
    }
  }, [localParticipant, isMicrophoneEnabled]);

  // Keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'm' || e.key === 'M') {
        if (!e.metaKey && !e.ctrlKey && !e.altKey) {
          e.preventDefault();
          toggleMic();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [toggleMic]);

  return (
    <button
      onClick={toggleMic}
      disabled={isPending}
      className={cn(
        'btn-hud flex items-center justify-center gap-2 transition-all',
        isMicrophoneEnabled
          ? 'border-terminator-cyan text-terminator-cyan hover:bg-terminator-cyan/10'
          : 'border-terminator-red text-terminator-red hover:bg-terminator-red/10'
      )}
    >
      {isPending ? (
        <Loader2 className="w-4 h-4 animate-spin" />
      ) : isMicrophoneEnabled ? (
        <Mic className="w-4 h-4" />
      ) : (
        <MicOff className="w-4 h-4" />
      )}
      <span>{isMicrophoneEnabled ? 'Mic On' : 'Mic Off'}</span>
    </button>
  );
};

/**
 * Standalone control panel for use outside LiveKitRoom context
 * (used when not connected)
 */
export const StandaloneControlPanel = ({
  connectionState,
  onConnect,
  onDisconnect,
  className,
}: ControlPanelProps) => {
  const isConnected = connectionState === 'connected';
  const isConnecting = connectionState === 'connecting' || connectionState === 'reconnecting';
  const canConnect = connectionState === 'disconnected' || connectionState === 'error';

  // Keyboard shortcut for ESC to disconnect
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isConnected) {
        onDisconnect();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isConnected, onDisconnect]);

  return (
    <div
      className={cn(
        'hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-4',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-4 pb-2 border-b border-terminator-border">
        <span className="text-terminator-red text-xs font-mono tracking-wider">
          ▸ CONTROLS
        </span>
        <span className="flex-1" />
        <span
          className={cn(
            'text-xs font-mono tracking-wider px-2 py-0.5 rounded',
            isConnected && 'bg-terminator-cyan/20 text-terminator-cyan',
            isConnecting && 'bg-yellow-500/20 text-yellow-400',
            !isConnected && !isConnecting && 'bg-terminator-border text-terminator-text-dim'
          )}
        >
          {connectionState.toUpperCase()}
        </span>
      </div>

      {/* Main Controls */}
      <div className="space-y-3">
        {canConnect ? (
          <button
            onClick={onConnect}
            className={cn(
              'w-full btn-hud-primary flex items-center justify-center gap-2',
              'group relative overflow-hidden'
            )}
          >
            <Phone className="w-4 h-4 transition-transform group-hover:scale-110" />
            <span>Initialize Connection</span>
          </button>
        ) : isConnecting ? (
          <button
            disabled
            className="w-full btn-hud flex items-center justify-center gap-2 cursor-wait"
          >
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Establishing Link...</span>
          </button>
        ) : (
          <div className="grid grid-cols-2 gap-2">
            <button disabled className="btn-hud flex items-center justify-center gap-2 opacity-50">
              <Mic className="w-4 h-4" />
              <span>Mic On</span>
            </button>
            <button
              onClick={onDisconnect}
              className="btn-hud-danger flex items-center justify-center gap-2"
            >
              <PhoneOff className="w-4 h-4" />
              <span>End Call</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
