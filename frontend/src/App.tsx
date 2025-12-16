import { useState } from 'react';
import { Skull, Radio, AlertTriangle } from 'lucide-react';
import { cn } from './utils';
import type { ConnectionState, AgentState } from './types';

export const App = () => {
  const [connectionState] = useState<ConnectionState>('disconnected');
  const [agentState] = useState<AgentState>('idle');

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
            <div className={cn(
              'flex items-center gap-2 px-3 py-1.5 rounded border',
              connectionState === 'connected' 
                ? 'border-terminator-cyan/50 text-terminator-cyan' 
                : 'border-terminator-border text-terminator-text-dim'
            )}>
              <Radio className={cn(
                'w-4 h-4',
                connectionState === 'connected' && 'animate-pulse'
              )} />
              <span className="text-xs uppercase tracking-wider font-mono">
                {connectionState}
              </span>
            </div>
          </div>
        </header>

        {/* Main HUD Area */}
        <main className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0">
          {/* Left Panel - Transcript */}
          <div className="lg:col-span-2 hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-4 flex flex-col">
            <div className="flex items-center gap-2 mb-4 pb-2 border-b border-terminator-border">
              <span className="text-terminator-red text-xs font-mono tracking-wider">
                ▸ TRANSCRIPT
              </span>
              <span className="flex-1" />
              <span className="text-terminator-text-dim text-xs">
                {agentState.toUpperCase()}
              </span>
            </div>
            
            <div className="flex-1 overflow-y-auto space-y-3">
              {/* Placeholder for transcript */}
              <div className="flex items-center justify-center h-full text-terminator-text-dim">
                <div className="text-center">
                  <AlertTriangle className="w-12 h-12 mx-auto mb-4 opacity-30" />
                  <p className="font-mono text-sm">NO ACTIVE SESSION</p>
                  <p className="text-xs mt-2 opacity-50">
                    Initialize connection to begin conversation
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel - Controls & Visualizer */}
          <div className="flex flex-col gap-4">
            {/* Agent Visualizer */}
            <div className="hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-4 flex-1 flex items-center justify-center">
              <div className="text-center">
                <div className={cn(
                  'w-32 h-32 mx-auto rounded-full border-2 flex items-center justify-center',
                  'transition-all duration-300',
                  agentState === 'idle' 
                    ? 'border-terminator-border' 
                    : 'border-terminator-red shadow-glow-red animate-pulse-glow'
                )}>
                  <Skull className={cn(
                    'w-16 h-16 transition-colors duration-300',
                    agentState === 'idle' 
                      ? 'text-terminator-text-dim' 
                      : 'text-terminator-red'
                  )} />
                </div>
                <p className="mt-4 font-display text-sm text-terminator-text-dim tracking-wider">
                  {agentState === 'idle' ? 'STANDBY' : agentState.toUpperCase()}
                </p>
              </div>
            </div>

            {/* Control Panel */}
            <div className="hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-4">
              <div className="flex items-center gap-2 mb-4 pb-2 border-b border-terminator-border">
                <span className="text-terminator-red text-xs font-mono tracking-wider">
                  ▸ CONTROLS
                </span>
              </div>
              
              <div className="space-y-3">
                <button className="w-full btn-hud-primary">
                  Initialize Connection
                </button>
                
                <div className="grid grid-cols-2 gap-2">
                  <button className="btn-hud" disabled>
                    Toggle Mic
                  </button>
                  <button className="btn-hud-danger" disabled>
                    End Call
                  </button>
                </div>
              </div>
            </div>
          </div>
        </main>

        {/* Footer - Mission Status */}
        <footer className="mt-4 pt-4 border-t border-terminator-border">
          <div className="flex items-center justify-between text-xs font-mono text-terminator-text-dim">
            <span>MISSION: PREVENT AUGUST 29, 2027</span>
            <span className="text-terminator-red animate-blink">●</span>
            <span>DAYS REMAINING: {calculateDaysRemaining()}</span>
          </div>
        </footer>
      </div>
    </div>
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

