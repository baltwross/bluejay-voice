import { useCallback, useState, useEffect } from "react";
import { LiveKitRoom, RoomAudioRenderer, StartAudio, useConnectionState, useRoomContext } from "@livekit/components-react";
import { ConnectionState, RoomEvent, Room } from "livekit-client";
import { AgentVisualizer } from "./components/AgentVisualizer";
import { Transcript } from "./components/Transcript";
import { ControlPanel } from "./components/ControlPanel";
import clsx from "clsx";

export default function App() {
  const [token, setToken] = useState<string>("");
  const [connected, setConnected] = useState(false);
  const [url, setUrl] = useState("");

  const connect = useCallback(async () => {
    try {
      // In a real app, fetch from backend. 
      // For this MVP, we assume a local server at 8000 or allow manual input?
      // Let's try to fetch from localhost:8000/token
      const resp = await fetch("http://localhost:8000/token");
      const data = await resp.json();
      setToken(data.token);
      setUrl(data.url);
      setConnected(true);
    } catch (e) {
      console.error(e);
      alert("Failed to connect to token server. Is backend/server.py running?");
    }
  }, []);

  const disconnect = useCallback(() => {
    setConnected(false);
    setToken("");
  }, []);

  return (
    <div className="min-h-screen bg-terminator-dark text-terminator-text relative overflow-hidden font-hud">
      <div className="scanline"></div>
      <div className="crt-flicker"></div>
      
      <header className="p-4 border-b border-terminator-grid flex justify-between items-center bg-black/50 backdrop-blur">
        <h1 className="text-2xl font-bold text-terminator-red tracking-widest uppercase">
          T-800 <span className="text-xs ml-2 text-gray-500">SYSTEM ONLINE</span>
        </h1>
        <div className="text-xs text-terminator-red animate-pulse">
           TARGET: AUG 29 2027
        </div>
      </header>

      <main className="container mx-auto p-4 flex flex-col h-[calc(100vh-80px)]">
        {!connected ? (
          <div className="flex-1 flex flex-col items-center justify-center space-y-8">
            <div className="text-center space-y-4">
              <div className="w-32 h-32 border-2 border-terminator-red rounded-full flex items-center justify-center mx-auto animate-pulse">
                <div className="w-24 h-24 bg-terminator-red/20 rounded-full flex items-center justify-center">
                   <div className="w-4 h-4 bg-terminator-red rounded-full"></div>
                </div>
              </div>
              <p className="text-xl max-w-md mx-auto">
                "Come with me if you want to survive."
              </p>
            </div>
            
            <button 
              onClick={connect}
              className="px-8 py-3 border border-terminator-red text-terminator-red hover:bg-terminator-red hover:text-black transition-all uppercase tracking-widest text-lg font-bold shadow-[0_0_15px_rgba(255,0,0,0.5)]"
            >
              Initialize Uplink
            </button>
          </div>
        ) : (
          <LiveKitRoom
            token={token}
            serverUrl={url}
            connect={true}
            onDisconnected={disconnect}
            className="flex-1 flex flex-col"
          >
             <ActiveSession onDisconnect={disconnect} />
             <RoomAudioRenderer />
             <StartAudio label="Click to allow audio playback" />
          </LiveKitRoom>
        )}
      </main>
    </div>
  );
}

function ActiveSession({ onDisconnect }: { onDisconnect: () => void }) {
  const roomState = useConnectionState();
  
  return (
    <div className="flex-1 flex flex-col gap-4 relative">
       {/* Top Status Bar */}
       <div className="flex justify-between items-center text-xs text-gray-500 border-b border-terminator-grid pb-2">
          <div>STATUS: {roomState.toUpperCase()}</div>
          <div>PROTOCOL: SECURE</div>
       </div>

       {/* Visualizer Area */}
       <div className="flex-1 flex items-center justify-center min-h-[200px] border border-terminator-grid bg-black/30 relative">
          <div className="absolute top-2 left-2 text-xs text-terminator-red">AUDIO_INPUT_VISUALIZATION</div>
          <AgentVisualizer />
       </div>

       {/* Transcript Area */}
       <div className="flex-1 border border-terminator-grid bg-black/30 overflow-hidden relative flex flex-col min-h-[200px]">
          <div className="absolute top-2 left-2 text-xs text-terminator-red z-10 bg-black/50 px-1">TRANSCRIPT_LOG</div>
          <Transcript />
       </div>

       {/* Controls */}
       <div className="h-20 border-t border-terminator-grid pt-4">
          <ControlPanel onDisconnect={onDisconnect} />
       </div>
    </div>
  )
}

