import { useState, useCallback, useEffect, useRef } from 'react';
import { Room, RoomEvent, ConnectionState as LiveKitConnectionState, Participant, ParticipantKind } from 'livekit-client';
import type { ConnectionState, AgentState } from '../types';

interface UseAgentOptions {
  /** LiveKit server URL */
  serverUrl?: string;
  /** Callback when connection state changes */
  onConnectionStateChange?: (state: ConnectionState) => void;
  /** Callback when agent state changes */
  onAgentStateChange?: (state: AgentState) => void;
}

interface UseAgentReturn {
  /** Current connection state */
  connectionState: ConnectionState;
  /** Current agent state */
  agentState: AgentState;
  /** LiveKit Room instance */
  room: Room | null;
  /** Token for connecting */
  token: string | null;
  /** Connect to the room */
  connect: () => Promise<void>;
  /** Disconnect from the room */
  disconnect: () => void;
  /** Whether the agent participant is present */
  isAgentConnected: boolean;
  /** Error message if any */
  error: string | null;
}

/** Map LiveKit connection state to our internal state */
function mapConnectionState(livekitState: LiveKitConnectionState): ConnectionState {
  switch (livekitState) {
    case LiveKitConnectionState.Connected:
      return 'connected';
    case LiveKitConnectionState.Connecting:
      return 'connecting';
    case LiveKitConnectionState.Reconnecting:
      return 'reconnecting';
    case LiveKitConnectionState.Disconnected:
      return 'disconnected';
    default:
      return 'disconnected';
  }
}

/** Parse agent state from participant attributes */
function parseAgentState(participant: Participant | undefined): AgentState {
  if (!participant) return 'idle';
  
  const stateAttr = participant.attributes?.['lk.agent.state'];
  switch (stateAttr) {
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

/**
 * useAgent - Hook for managing LiveKit agent connection
 * 
 * Handles:
 * - Token fetching
 * - Room connection/disconnection
 * - Connection state tracking
 * - Agent participant detection
 * - Agent state monitoring
 */
export function useAgent(options: UseAgentOptions = {}): UseAgentReturn {
  const { serverUrl, onConnectionStateChange, onAgentStateChange } = options;
  
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [agentState, setAgentState] = useState<AgentState>('idle');
  const [room, setRoom] = useState<Room | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isAgentConnected, setIsAgentConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const roomRef = useRef<Room | null>(null);

  // Update connection state and notify
  const updateConnectionState = useCallback((state: ConnectionState) => {
    setConnectionState(state);
    onConnectionStateChange?.(state);
  }, [onConnectionStateChange]);

  // Update agent state and notify
  const updateAgentState = useCallback((state: AgentState) => {
    setAgentState(state);
    onAgentStateChange?.(state);
  }, [onAgentStateChange]);

  // Find agent participant in the room
  const findAgentParticipant = useCallback((room: Room): Participant | undefined => {
    for (const participant of room.remoteParticipants.values()) {
      if (participant.kind === ParticipantKind.AGENT) {
        return participant;
      }
    }
    return undefined;
  }, []);

  // Fetch token from the backend
  const fetchToken = useCallback(async (): Promise<{ token: string; roomName: string; serverUrl: string }> => {
    const response = await fetch('/api/token');
    if (!response.ok) {
      throw new Error('Failed to fetch token');
    }
    return response.json();
  }, []);

  // Connect to the room
  const connect = useCallback(async () => {
    if (connectionState === 'connected' || connectionState === 'connecting') {
      console.warn('Already connected or connecting');
      return;
    }

    setError(null);
    updateConnectionState('connecting');

    try {
      // Fetch token
      const tokenData = await fetchToken();
      setToken(tokenData.token);

      // Create room
      const newRoom = new Room({
        adaptiveStream: true,
        dynacast: true,
      });
      
      roomRef.current = newRoom;
      setRoom(newRoom);

      // Setup event listeners
      newRoom.on(RoomEvent.ConnectionStateChanged, (state) => {
        updateConnectionState(mapConnectionState(state));
      });

      newRoom.on(RoomEvent.ParticipantConnected, (participant) => {
        if (participant.kind === ParticipantKind.AGENT) {
          setIsAgentConnected(true);
          updateAgentState(parseAgentState(participant));
        }
      });

      newRoom.on(RoomEvent.ParticipantDisconnected, (participant) => {
        if (participant.kind === ParticipantKind.AGENT) {
          setIsAgentConnected(false);
          updateAgentState('idle');
        }
      });

      newRoom.on(RoomEvent.ParticipantAttributesChanged, (_, participant) => {
        if (participant.kind === ParticipantKind.AGENT) {
          updateAgentState(parseAgentState(participant));
        }
      });

      newRoom.on(RoomEvent.Disconnected, () => {
        updateConnectionState('disconnected');
        setIsAgentConnected(false);
        updateAgentState('idle');
      });

      // Connect to room
      await newRoom.connect(
        tokenData.serverUrl || serverUrl || '',
        tokenData.token,
        {
          autoSubscribe: true,
        }
      );

      // Enable microphone
      await newRoom.localParticipant.setMicrophoneEnabled(true);

      // Check if agent is already in the room
      const agentParticipant = findAgentParticipant(newRoom);
      if (agentParticipant) {
        setIsAgentConnected(true);
        updateAgentState(parseAgentState(agentParticipant));
      }

      updateConnectionState('connected');
    } catch (err) {
      console.error('Failed to connect:', err);
      setError(err instanceof Error ? err.message : 'Failed to connect');
      updateConnectionState('error');
      
      // Cleanup on error
      if (roomRef.current) {
        roomRef.current.disconnect();
        roomRef.current = null;
        setRoom(null);
      }
    }
  }, [connectionState, fetchToken, serverUrl, findAgentParticipant, updateConnectionState, updateAgentState]);

  // Disconnect from the room
  const disconnect = useCallback(() => {
    if (roomRef.current) {
      roomRef.current.disconnect();
      roomRef.current = null;
      setRoom(null);
    }
    setToken(null);
    setIsAgentConnected(false);
    updateConnectionState('disconnected');
    updateAgentState('idle');
  }, [updateConnectionState, updateAgentState]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (roomRef.current) {
        roomRef.current.disconnect();
      }
    };
  }, []);

  return {
    connectionState,
    agentState,
    room,
    token,
    connect,
    disconnect,
    isAgentConnected,
    error,
  };
}


