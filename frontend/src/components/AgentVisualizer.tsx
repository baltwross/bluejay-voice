import { BarVisualizer, useTracks, useLocalParticipant } from "@livekit/components-react";
import { Track } from "livekit-client";

export function AgentVisualizer() {
  // Get remote audio tracks (the agent)
  const tracks = useTracks([Track.Source.Microphone, Track.Source.Unknown]); // Agent might use Unknown source
  const { localParticipant } = useLocalParticipant();
  
  // Find a remote track
  const agentTrack = tracks.find(t => t.participant.identity !== localParticipant.identity);

  return (
    <div className="w-full h-full flex items-center justify-center p-8">
      {agentTrack ? (
        <BarVisualizer
          state={agentTrack}
          barCount={30}
          trackRef={agentTrack}
          className="h-32 w-full"
          options={{ color: '#ff0000', thickness: 4, gap: 4 }} 
        />
      ) : (
        <div className="text-terminator-grid animate-pulse text-4xl font-bold opacity-20">
          AWAITING_SIGNAL
        </div>
      )}
    </div>
  );
}

