import { TrackToggle, useDisconnect, useLocalParticipant } from "@livekit/components-react";
import { Track } from "livekit-client";
import { useState } from "react";
import { Mic, MicOff, PhoneOff, Send } from "lucide-react";

export function ControlPanel({ onDisconnect }: { onDisconnect: () => void }) {
  const { disconnect } = useDisconnect();
  const [inputText, setInputText] = useState("");
  const { localParticipant } = useLocalParticipant();

  const handleDisconnect = () => {
    disconnect();
    onDisconnect();
  };

  const handleSend = async () => {
    if (!inputText.trim()) return;
    
    if (localParticipant) {
      // Send data packet for URL/Text
      const encoder = new TextEncoder();
      const data = encoder.encode(JSON.stringify({ type: "USER_INPUT", content: inputText }));
      await localParticipant.publishData(data, { reliable: true });
      setInputText("");
    }
  };

  return (
    <div className="flex items-center gap-4 h-full px-4">
      {/* Mic Toggle */}
      <TrackToggle source={Track.Source.Microphone} className="p-3 border border-terminator-red rounded-full hover:bg-terminator-red/20 transition-colors text-terminator-red">
         {(enabled) => enabled ? <Mic /> : <MicOff />}
      </TrackToggle>

      {/* Input Console */}
      <div className="flex-1 flex gap-2">
        <input 
          type="text" 
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="ENTER_COMMAND_OR_URL..." 
          className="flex-1 bg-black border border-terminator-grid text-terminator-red px-4 py-2 focus:outline-none focus:border-terminator-red font-mono text-sm placeholder-terminator-red/30"
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
        />
        <button 
          onClick={handleSend}
          className="p-2 border border-terminator-red text-terminator-red hover:bg-terminator-red hover:text-black transition-colors"
        >
          <Send size={18} />
        </button>
      </div>

      {/* End Call */}
      <button 
        onClick={handleDisconnect}
        className="p-3 border border-red-600 bg-red-900/20 rounded-full hover:bg-red-600 hover:text-white transition-colors text-red-500"
      >
        <PhoneOff />
      </button>
    </div>
  );
}

