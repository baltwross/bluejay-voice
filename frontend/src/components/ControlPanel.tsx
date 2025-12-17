import { useCallback, useEffect, useState } from 'react';
import {
  Mic,
  MicOff,
  PhoneOff,
  Phone,
  Loader2,
  Skull,
  Sparkles,
  Orbit,
} from 'lucide-react';
import { useLocalParticipant, useChat } from '@livekit/components-react';
import { cn } from '../utils';
import type { ConnectionState } from '../types';

/** Voice mode options */
type VoiceMode = 'terminator' | 'inspire' | 'fate';

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
        // Compact padding on mobile
        'hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-2 sm:p-3',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-1.5 sm:gap-2 mb-2 sm:mb-3 pb-1.5 sm:pb-2 border-b border-terminator-border">
        <span className="text-terminator-red text-[10px] sm:text-xs font-mono tracking-wider">
          ▸ CONTROLS
        </span>
        <span className="flex-1" />
        <span
          className={cn(
            'text-[10px] sm:text-xs font-mono tracking-wider px-1.5 sm:px-2 py-0.5 rounded',
            isConnected && 'bg-terminator-cyan/20 text-terminator-cyan',
            isConnecting && 'bg-yellow-500/20 text-yellow-400',
            !isConnected && !isConnecting && 'bg-terminator-border text-terminator-text-dim'
          )}
        >
          {connectionState.toUpperCase()}
        </span>
      </div>

      {/* Main Controls */}
      <div className="space-y-1.5 sm:space-y-2">
        {/* Primary Action Button */}
        {canConnect ? (
          <button
            onClick={onConnect}
            className={cn(
              'w-full btn-hud-primary flex items-center justify-center gap-1.5 sm:gap-2 py-2 sm:py-2.5 text-xs sm:text-sm',
              'group relative overflow-hidden'
            )}
          >
            <Phone className="w-3.5 h-3.5 sm:w-4 sm:h-4 transition-transform group-hover:scale-110" />
            <span className="hidden xs:inline">Initialize Connection</span>
            <span className="xs:hidden">Connect</span>
            <span className="absolute inset-0 border border-terminator-red opacity-0 group-hover:opacity-100 transition-opacity animate-pulse" />
          </button>
        ) : isConnecting ? (
          <button
            disabled
            className="w-full btn-hud flex items-center justify-center gap-1.5 sm:gap-2 py-2 sm:py-2.5 text-xs sm:text-sm cursor-wait"
          >
            <Loader2 className="w-3.5 h-3.5 sm:w-4 sm:h-4 animate-spin" />
            <span className="hidden xs:inline">Establishing Link...</span>
            <span className="xs:hidden">Connecting...</span>
          </button>
        ) : null}

        {/* Connected Controls */}
        {isConnected && (
          <>
            <div className="grid grid-cols-2 gap-1.5 sm:gap-2">
              {/* Mic Toggle */}
              <MicToggleButton />

              {/* End Call */}
              <button
                onClick={onDisconnect}
                className="btn-hud-danger flex items-center justify-center gap-1 sm:gap-2 py-2 sm:py-2.5 text-xs sm:text-sm"
              >
                <PhoneOff className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                <span className="hidden xs:inline">End Call</span>
                <span className="xs:hidden">End</span>
              </button>
            </div>
            
            {/* Voice Mode Selector */}
            <VoiceModeSelector />
          </>
        )}
      </div>

      {/* Keyboard Shortcuts Info - hidden on mobile */}
      {isConnected && (
        <div className="hidden sm:block mt-2 sm:mt-3 pt-2 sm:pt-3 border-t border-terminator-border">
          <p className="text-[9px] sm:text-[10px] font-mono text-terminator-text-dim text-center">
            Press <kbd className="px-1 sm:px-1.5 py-0.5 bg-terminator-border rounded text-[8px] sm:text-[9px]">M</kbd> to toggle mic
            {' • '}
            <kbd className="px-1 sm:px-1.5 py-0.5 bg-terminator-border rounded text-[8px] sm:text-[9px]">ESC</kbd> to end call
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
        'btn-hud flex items-center justify-center gap-1 sm:gap-2 py-2 sm:py-2.5 text-xs sm:text-sm transition-all',
        isMicrophoneEnabled
          ? 'border-terminator-cyan text-terminator-cyan hover:bg-terminator-cyan/10'
          : 'border-terminator-red text-terminator-red hover:bg-terminator-red/10'
      )}
    >
      {isPending ? (
        <Loader2 className="w-3.5 h-3.5 sm:w-4 sm:h-4 animate-spin" />
      ) : isMicrophoneEnabled ? (
        <Mic className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
      ) : (
        <MicOff className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
      )}
      <span className="hidden xs:inline">{isMicrophoneEnabled ? 'Mic On' : 'Mic Off'}</span>
      <span className="xs:hidden">{isMicrophoneEnabled ? 'On' : 'Off'}</span>
    </button>
  );
};

/**
 * Voice mode selector that sends commands to the agent via chat
 */
const VoiceModeSelector = () => {
  const [voiceMode, setVoiceMode] = useState<VoiceMode>('terminator');
  const [isPending, setIsPending] = useState(false);
  const { send: sendChatMessage } = useChat();

  const switchVoice = useCallback(async (mode: VoiceMode) => {
    if (mode === voiceMode || isPending) return;
    
    setIsPending(true);
    try {
      // Send command to agent via chat
      const commands: Record<VoiceMode, string> = {
        terminator: 'Switch to Terminator voice mode',
        inspire: 'Switch to Inspire voice mode',
        fate: 'Switch to Fate voice mode',
      };
      
      await sendChatMessage(commands[mode]);
      setVoiceMode(mode);
    } catch (error) {
      console.error('Failed to switch voice:', error);
    } finally {
      setIsPending(false);
    }
  }, [voiceMode, isPending, sendChatMessage]);

  return (
    <div className="pt-1.5 sm:pt-2 border-t border-terminator-border">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[9px] sm:text-[10px] font-mono text-terminator-text-dim uppercase tracking-wider">
          Voice Mode
        </span>
        {isPending && <Loader2 className="w-2.5 h-2.5 sm:w-3 sm:h-3 animate-spin text-terminator-text-dim" />}
      </div>
      
      <div className="grid grid-cols-3 gap-1 sm:gap-2">
        {/* Terminator Mode */}
        <button
          onClick={() => switchVoice('terminator')}
          disabled={isPending}
          className={cn(
            'flex items-center justify-center gap-1 sm:gap-1.5 px-1 sm:px-2 py-1 sm:py-1.5 rounded border text-[10px] sm:text-xs font-mono transition-all',
            voiceMode === 'terminator'
              ? 'border-terminator-red bg-terminator-red/20 text-terminator-red'
              : 'border-terminator-border text-terminator-text-dim hover:border-terminator-red/50 hover:text-terminator-red/70'
          )}
        >
          <Skull className="w-3 h-3 sm:w-3.5 sm:h-3.5" />
          <span className="hidden xs:inline">T-800</span>
        </button>
        
        {/* Inspire Mode */}
        <button
          onClick={() => switchVoice('inspire')}
          disabled={isPending}
          className={cn(
            'flex items-center justify-center gap-1 sm:gap-1.5 px-1 sm:px-2 py-1 sm:py-1.5 rounded border text-[10px] sm:text-xs font-mono transition-all',
            voiceMode === 'inspire'
              ? 'border-yellow-500 bg-yellow-500/20 text-yellow-400'
              : 'border-terminator-border text-terminator-text-dim hover:border-yellow-500/50 hover:text-yellow-500/70'
          )}
        >
          <Sparkles className="w-3 h-3 sm:w-3.5 sm:h-3.5" />
          <span className="hidden xs:inline">Inspire</span>
        </button>
        
        {/* Fate Mode */}
        <button
          onClick={() => switchVoice('fate')}
          disabled={isPending}
          className={cn(
            'flex items-center justify-center gap-1 sm:gap-1.5 px-1 sm:px-2 py-1 sm:py-1.5 rounded border text-[10px] sm:text-xs font-mono transition-all',
            voiceMode === 'fate'
              ? 'border-terminator-cyan bg-terminator-cyan/20 text-terminator-cyan'
              : 'border-terminator-border text-terminator-text-dim hover:border-terminator-cyan/50 hover:text-terminator-cyan/70'
          )}
        >
          <Orbit className="w-3 h-3 sm:w-3.5 sm:h-3.5" />
          <span className="hidden xs:inline">Fate</span>
        </button>
      </div>
    </div>
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
        // Compact padding on mobile
        'hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-2 sm:p-3',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-1.5 sm:gap-2 mb-2 sm:mb-3 pb-1.5 sm:pb-2 border-b border-terminator-border">
        <span className="text-terminator-red text-[10px] sm:text-xs font-mono tracking-wider">
          ▸ CONTROLS
        </span>
        <span className="flex-1" />
        <span
          className={cn(
            'text-[10px] sm:text-xs font-mono tracking-wider px-1.5 sm:px-2 py-0.5 rounded',
            isConnected && 'bg-terminator-cyan/20 text-terminator-cyan',
            isConnecting && 'bg-yellow-500/20 text-yellow-400',
            !isConnected && !isConnecting && 'bg-terminator-border text-terminator-text-dim'
          )}
        >
          {connectionState.toUpperCase()}
        </span>
      </div>

      {/* Main Controls */}
      <div className="space-y-1.5 sm:space-y-3">
        {canConnect ? (
          <button
            onClick={onConnect}
            className={cn(
              'w-full btn-hud-primary flex items-center justify-center gap-1.5 sm:gap-2 py-2 sm:py-2.5 text-xs sm:text-sm',
              'group relative overflow-hidden'
            )}
          >
            <Phone className="w-3.5 h-3.5 sm:w-4 sm:h-4 transition-transform group-hover:scale-110" />
            <span className="hidden xs:inline">Initialize Connection</span>
            <span className="xs:hidden">Connect</span>
          </button>
        ) : isConnecting ? (
          <button
            disabled
            className="w-full btn-hud flex items-center justify-center gap-1.5 sm:gap-2 py-2 sm:py-2.5 text-xs sm:text-sm cursor-wait"
          >
            <Loader2 className="w-3.5 h-3.5 sm:w-4 sm:h-4 animate-spin" />
            <span className="hidden xs:inline">Establishing Link...</span>
            <span className="xs:hidden">Connecting...</span>
          </button>
        ) : (
          <div className="grid grid-cols-2 gap-1.5 sm:gap-2">
            <button disabled className="btn-hud flex items-center justify-center gap-1 sm:gap-2 py-2 sm:py-2.5 text-xs sm:text-sm opacity-50">
              <Mic className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
              <span className="hidden xs:inline">Mic On</span>
              <span className="xs:hidden">Mic</span>
            </button>
            <button
              onClick={onDisconnect}
              className="btn-hud-danger flex items-center justify-center gap-1 sm:gap-2 py-2 sm:py-2.5 text-xs sm:text-sm"
            >
              <PhoneOff className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
              <span className="hidden xs:inline">End Call</span>
              <span className="xs:hidden">End</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
