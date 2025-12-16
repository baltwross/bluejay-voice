import { useEffect, useRef } from 'react';
import { Skull, Volume2, Mic, Brain, Radio } from 'lucide-react';
import { useVoiceAssistant, BarVisualizer } from '@livekit/components-react';
import { cn } from '../utils';
import type { AgentState } from '../types';

interface AgentVisualizerProps {
  /** Custom class name for the container */
  className?: string;
}

/** Map LiveKit agent state to our internal state */
function mapAgentState(livekitState: string | undefined): AgentState {
  switch (livekitState) {
    case 'listening':
      return 'listening';
    case 'thinking':
      return 'thinking';
    case 'speaking':
      return 'speaking';
    default:
      return 'idle';
  }
}

/** Get the appropriate icon for the current state */
function getStateIcon(state: AgentState) {
  switch (state) {
    case 'listening':
      return <Mic className="w-6 h-6" />;
    case 'thinking':
      return <Brain className="w-6 h-6" />;
    case 'speaking':
      return <Volume2 className="w-6 h-6" />;
    case 'reading':
      return <Radio className="w-6 h-6" />;
    default:
      return <Skull className="w-6 h-6" />;
  }
}

/** Get status text for display */
function getStatusText(state: AgentState): string {
  switch (state) {
    case 'listening':
      return 'RECEIVING AUDIO INPUT';
    case 'thinking':
      return 'PROCESSING...';
    case 'speaking':
      return 'TRANSMITTING';
    case 'reading':
      return 'READING DOCUMENT';
    default:
      return 'STANDBY';
  }
}

/**
 * AgentVisualizer - Displays the T-800 voice agent's audio visualization
 * 
 * Uses LiveKit's BarVisualizer for audio waveform display with a
 * futuristic Terminator HUD aesthetic.
 */
export const AgentVisualizer = ({ className }: AgentVisualizerProps) => {
  const { state: livekitState, audioTrack } = useVoiceAssistant();
  const state = mapAgentState(livekitState);
  const canvasRef = useRef<HTMLDivElement>(null);

  // Determine if agent is active (not idle)
  const isActive = state !== 'idle';
  const isSpeaking = state === 'speaking';
  const isThinking = state === 'thinking';

  return (
    <div
      className={cn(
        'hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-4',
        'flex flex-col items-center justify-center',
        'transition-all duration-300',
        className
      )}
    >
      {/* Header */}
      <div className="w-full flex items-center gap-2 mb-4 pb-2 border-b border-terminator-border">
        <span className="text-terminator-red text-xs font-mono tracking-wider">
          ▸ T-800 NEURAL NET
        </span>
        <span className="flex-1" />
        <span className={cn(
          'text-xs font-mono tracking-wider transition-colors',
          isActive ? 'text-terminator-cyan' : 'text-terminator-text-dim'
        )}>
          {state.toUpperCase()}
        </span>
      </div>

      {/* Main Visualizer Area */}
      <div className="flex-1 w-full flex flex-col items-center justify-center min-h-[200px]">
        {/* Skull Icon with Glow */}
        <div
          className={cn(
            'relative w-24 h-24 md:w-32 md:h-32 rounded-full border-2 flex items-center justify-center mb-4',
            'transition-all duration-500',
            isActive
              ? 'border-terminator-red shadow-glow-red'
              : 'border-terminator-border',
            isSpeaking && 'animate-pulse-glow',
            isThinking && 'animate-pulse'
          )}
        >
          {/* Animated ring when active */}
          {isActive && (
            <div className="absolute inset-0 rounded-full border border-terminator-red/30 animate-ping" />
          )}
          
          <Skull
            className={cn(
              'w-12 h-12 md:w-16 md:h-16 transition-all duration-300',
              isActive ? 'text-terminator-red' : 'text-terminator-text-dim'
            )}
          />
        </div>

        {/* Audio Visualizer Bars */}
        <div ref={canvasRef} className="w-full max-w-xs h-16 flex items-center justify-center">
          {audioTrack ? (
            <BarVisualizer
              state={livekitState}
              barCount={7}
              trackRef={audioTrack}
              className="w-full h-full"
              style={{
                '--lk-bar-color': isSpeaking ? 'var(--color-red)' : 'var(--color-cyan)',
              } as React.CSSProperties}
            />
          ) : (
            // Fallback static bars when no audio track
            <div className="flex items-center justify-center gap-1 h-full">
              {[...Array(7)].map((_, i) => (
                <div
                  key={i}
                  className={cn(
                    'w-2 rounded-sm transition-all duration-300',
                    isActive ? 'bg-terminator-red' : 'bg-terminator-border'
                  )}
                  style={{
                    height: isActive ? `${20 + Math.random() * 30}%` : '20%',
                    animationDelay: `${i * 100}ms`,
                  }}
                />
              ))}
            </div>
          )}
        </div>

        {/* Status Text */}
        <div className="mt-4 flex items-center gap-2">
          <div className={cn(
            'p-1.5 rounded-full transition-colors',
            isActive ? 'bg-terminator-red/20 text-terminator-red' : 'bg-terminator-border text-terminator-text-dim'
          )}>
            {getStateIcon(state)}
          </div>
          <p className={cn(
            'font-mono text-xs tracking-wider transition-colors',
            isActive ? 'text-terminator-cyan text-glow-cyan' : 'text-terminator-text-dim'
          )}>
            {getStatusText(state)}
          </p>
        </div>
      </div>

      {/* Footer - Connection indicator */}
      <div className="w-full pt-2 mt-2 border-t border-terminator-border flex items-center justify-center">
        <div className="flex items-center gap-2">
          <div className={cn(
            'w-2 h-2 rounded-full transition-colors',
            isActive ? 'bg-terminator-cyan animate-pulse' : 'bg-terminator-text-dim'
          )} />
          <span className="text-[10px] font-mono text-terminator-text-dim tracking-widest">
            {isActive ? 'NEURAL NET ONLINE' : 'AWAITING CONNECTION'}
          </span>
        </div>
      </div>
    </div>
  );
};

