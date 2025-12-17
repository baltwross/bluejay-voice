import { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { cn } from '../utils';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

/**
 * ErrorBoundary - Catches JavaScript errors in child components
 * 
 * Provides a fallback UI when errors occur, preventing the entire
 * application from crashing.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  handleRetry = (): void => {
    this.setState({ hasError: false, error: null });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-[200px] flex items-center justify-center p-6">
          <div className="max-w-md w-full hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-full bg-terminator-red/20">
                <AlertTriangle className="w-6 h-6 text-terminator-red" />
              </div>
              <div>
                <h3 className="font-mono text-sm font-semibold text-terminator-red">
                  SYSTEM ERROR
                </h3>
                <p className="text-xs text-terminator-text-dim">
                  An unexpected error occurred
                </p>
              </div>
            </div>

            {this.state.error && (
              <div className="mb-4 p-3 bg-terminator-darker rounded text-xs font-mono text-terminator-text-dim overflow-auto max-h-32">
                {this.state.error.message}
              </div>
            )}

            <button
              onClick={this.handleRetry}
              className="w-full btn-hud-primary flex items-center justify-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Retry</span>
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * ConnectionError - Displayed when LiveKit connection fails
 */
interface ConnectionErrorProps {
  error: string;
  onRetry?: () => void;
}

export const ConnectionError = ({ error, onRetry }: ConnectionErrorProps) => {
  return (
    <div className="flex items-center justify-center p-4">
      <div className="max-w-sm w-full hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-4">
        <div className="flex items-center gap-3 mb-3">
          <AlertTriangle className="w-5 h-5 text-terminator-red" />
          <span className="font-mono text-sm text-terminator-red">
            CONNECTION FAILED
          </span>
        </div>
        
        <p className="text-xs font-mono text-terminator-text-dim mb-4">
          {error}
        </p>

        {onRetry && (
          <button
            onClick={onRetry}
            className="w-full btn-hud flex items-center justify-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Retry Connection</span>
          </button>
        )}
      </div>
    </div>
  );
};



