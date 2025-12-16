import { useState, useCallback, useEffect, useRef } from 'react';
import type { ConnectionState } from '../types';

interface TokenResponse {
  token: string;
  roomName: string;
  serverUrl: string;
}

interface UseConnectionOptions {
  /** Token endpoint URL */
  tokenEndpoint?: string;
  /** Whether to auto-reconnect on disconnect */
  autoReconnect?: boolean;
  /** Max reconnection attempts */
  maxReconnectAttempts?: number;
  /** Reconnection delay in ms */
  reconnectDelay?: number;
}

interface UseConnectionReturn {
  /** Current connection state */
  connectionState: ConnectionState;
  /** LiveKit access token */
  token: string | null;
  /** Room name */
  roomName: string | null;
  /** Server URL */
  serverUrl: string | null;
  /** Start connecting (fetch token) */
  connect: () => Promise<void>;
  /** Disconnect and cleanup */
  disconnect: () => void;
  /** Error message if any */
  error: string | null;
  /** Whether currently fetching token */
  isLoading: boolean;
}

/**
 * useConnection - Hook for managing LiveKit connection credentials
 * 
 * Handles:
 * - Token fetching from backend API
 * - Connection state management
 * - Auto-reconnection with exponential backoff
 * - Error handling
 */
export function useConnection(options: UseConnectionOptions = {}): UseConnectionReturn {
  const {
    tokenEndpoint = '/api/token',
    autoReconnect = false,
    maxReconnectAttempts = 3,
    reconnectDelay = 1000,
  } = options;

  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [token, setToken] = useState<string | null>(null);
  const [roomName, setRoomName] = useState<string | null>(null);
  const [serverUrl, setServerUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const reconnectAttempts = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup reconnect timeout
  const cleanupReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    reconnectAttempts.current = 0;
  }, []);

  // Fetch token from backend
  const fetchToken = useCallback(async (): Promise<TokenResponse> => {
    const response = await fetch(tokenEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `HTTP ${response.status}`);
    }

    return response.json();
  }, [tokenEndpoint]);

  // Connect (fetch token and prepare for connection)
  const connect = useCallback(async () => {
    if (isLoading) {
      console.warn('Already connecting');
      return;
    }

    setIsLoading(true);
    setError(null);
    setConnectionState('connecting');
    cleanupReconnect();

    try {
      const tokenData = await fetchToken();
      
      setToken(tokenData.token);
      setRoomName(tokenData.roomName);
      setServerUrl(tokenData.serverUrl);
      setConnectionState('connected');
      reconnectAttempts.current = 0;
    } catch (err) {
      console.error('Failed to fetch token:', err);
      const errorMessage = err instanceof Error ? err.message : 'Connection failed';
      setError(errorMessage);
      setConnectionState('error');

      // Auto-reconnect logic
      if (autoReconnect && reconnectAttempts.current < maxReconnectAttempts) {
        reconnectAttempts.current += 1;
        const delay = reconnectDelay * Math.pow(2, reconnectAttempts.current - 1);
        
        console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current}/${maxReconnectAttempts})`);
        setConnectionState('reconnecting');
        
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, delay);
      }
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, fetchToken, autoReconnect, maxReconnectAttempts, reconnectDelay, cleanupReconnect]);

  // Disconnect and cleanup
  const disconnect = useCallback(() => {
    cleanupReconnect();
    setToken(null);
    setRoomName(null);
    setServerUrl(null);
    setError(null);
    setConnectionState('disconnected');
  }, [cleanupReconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanupReconnect();
    };
  }, [cleanupReconnect]);

  return {
    connectionState,
    token,
    roomName,
    serverUrl,
    connect,
    disconnect,
    error,
    isLoading,
  };
}


