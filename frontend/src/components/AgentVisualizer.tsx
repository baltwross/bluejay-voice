import { useRef } from 'react';
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
      return <Mic className="w-4 h-4" />;
    case 'thinking':
      return <Brain className="w-4 h-4" />;
    case 'speaking':
      return <Volume2 className="w-4 h-4" />;
    case 'reading':
      return <Radio className="w-4 h-4" />;
    default:
      return <Skull className="w-4 h-4" />;
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
        // Denser padding on mobile, overflow-hidden so content stays within border
        'hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-2 sm:p-3',
        'flex flex-col items-center overflow-hidden',
        'transition-all duration-300',
        className
      )}
    >
      {/* Header */}
      <div className="w-full flex items-center gap-1.5 sm:gap-2 mb-1 sm:mb-2 pb-1 sm:pb-2 border-b border-terminator-border flex-shrink-0">
        <span className="text-terminator-red text-[8px] sm:text-[9px] font-mono tracking-wider">
          ▸ T-800 NEURAL NET
        </span>
        <span className="flex-1" />
        <span className={cn(
          'text-[10px] sm:text-xs font-mono tracking-wider transition-colors',
          isActive ? 'text-terminator-cyan' : 'text-terminator-text-dim'
        )}>
          {state.toUpperCase()}
        </span>
      </div>

      {/* Main Visualizer Area */}
      <div className="w-full flex flex-col items-center justify-center flex-1 min-h-0">
        {/* Skull Icon with Glow - smaller on mobile */}
        <div
          className={cn(
            'relative w-10 h-10 sm:w-14 sm:h-14 md:w-16 md:h-16 lg:w-20 lg:h-20 rounded-full border-2 flex items-center justify-center mb-1 sm:mb-2',
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
              'w-5 h-5 sm:w-7 sm:h-7 md:w-8 md:h-8 lg:w-10 lg:h-10 transition-all duration-300',
              isActive ? 'text-terminator-red' : 'text-terminator-text-dim'
            )}
          />
        </div>

        {/* Audio Visualizer Bars - shorter on mobile */}
        <div ref={canvasRef} className="w-full max-w-[200px] sm:max-w-xs h-6 sm:h-8 lg:h-10 flex items-center justify-center flex-shrink-0">
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
            <div className="flex items-center justify-center gap-0.5 sm:gap-1 h-full">
              {[...Array(7)].map((_, i) => (
                <div
                  key={i}
                  className={cn(
                    'w-1.5 sm:w-2 rounded-sm transition-all duration-300',
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

        {/* Status Text - hidden on very small screens */}
        <div className="mt-1 sm:mt-2 flex items-center gap-1.5 sm:gap-2 flex-shrink-0">
          <div className={cn(
            'p-0.5 sm:p-1 rounded-full transition-colors',
            isActive ? 'bg-terminator-red/20 text-terminator-red' : 'bg-terminator-border text-terminator-text-dim'
          )}>
            <div className="w-3 h-3 sm:w-4 sm:h-4 flex items-center justify-center">
              {getStateIcon(state)}
            </div>
          </div>
          <p className={cn(
            'font-mono text-[8px] sm:text-[10px] tracking-wider transition-colors',
            isActive ? 'text-terminator-cyan text-glow-cyan' : 'text-terminator-text-dim'
          )}>
            {getStatusText(state)}
          </p>
        </div>
      </div>

      {/* Footer - Connection indicator */}
      <div className="w-full pt-1 mt-1 border-t border-terminator-border flex items-center justify-center flex-shrink-0">
        <div className="flex items-center gap-1 sm:gap-2">
          <div className={cn(
            'w-1.5 h-1.5 sm:w-2 sm:h-2 rounded-full transition-colors',
            isActive ? 'bg-terminator-cyan animate-pulse' : 'bg-terminator-text-dim'
          )} />
          <span className="text-[8px] sm:text-[10px] font-mono text-terminator-text-dim tracking-widest">
            {isActive ? 'ONLINE' : 'STANDBY'}
          </span>
        </div>
      </div>
    </div>
  );
};

