import { useRoomContext } from "@livekit/components-react";
import { RoomEvent, Participant, TranscriptionSegment, TrackPublication, Track } from "livekit-client";
import { useEffect, useState, useRef } from "react";
import clsx from "clsx";

type TranscriptItem = {
  id: string;
  sender: string;
  text: string;
  isAgent: boolean;
  isFinal: boolean;
};

export function Transcript() {
  const room = useRoomContext();
  const [segments, setSegments] = useState<TranscriptItem[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!room) return;

    const handleTranscription = (
      transcriptions: TranscriptionSegment[],
      participant?: Participant,
      publication?: TrackPublication
    ) => {
      setSegments(prev => {
        const newSegments = [...prev];
        
        transcriptions.forEach(t => {
          const id = t.id;
          const existingIndex = newSegments.findIndex(s => s.id === id);
          
          const isAgent = participant?.identity?.includes("agent") || false; 
          // Note: Identity check is a heuristic. 
          
          const item: TranscriptItem = {
            id: t.id,
            sender: participant?.identity || "Unknown",
            text: t.text,
            isAgent: isAgent,
            isFinal: t.final,
          };

          if (existingIndex >= 0) {
            newSegments[existingIndex] = item;
          } else {
            newSegments.push(item);
          }
        });
        
        return newSegments.slice(-50); // Keep last 50
      });
    };

    room.on(RoomEvent.TranscriptionReceived, handleTranscription);
    return () => {
      room.off(RoomEvent.TranscriptionReceived, handleTranscription);
    };
  }, [room]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [segments]);

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-2 font-mono text-sm" ref={scrollRef}>
      {segments.length === 0 && (
         <div className="text-gray-600 italic">Listening for audio stream...</div>
      )}
      {segments.map((seg) => (
        <div key={seg.id} className={clsx(
            "flex gap-2",
            seg.isAgent ? "text-terminator-red" : "text-cyan-400"
        )}>
           <span className="font-bold uppercase text-xs opacity-50 w-16 shrink-0">
             {seg.isAgent ? "T-800" : "HUMAN"}
           </span>
           <span className={clsx(seg.isFinal ? "opacity-100" : "opacity-50")}>
             {seg.text}
           </span>
        </div>
      ))}
    </div>
  );
}

