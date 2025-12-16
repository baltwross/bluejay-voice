import { useState, useCallback, useRef, DragEvent, ChangeEvent } from 'react';
import {
  Upload,
  Link,
  FileText,
  Youtube,
  Globe,
  X,
  Loader2,
  CheckCircle,
  AlertCircle,
} from 'lucide-react';
import { cn } from '../utils';

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';
type SourceType = 'pdf' | 'youtube' | 'web' | 'file';

interface InputConsoleProps {
  /** Callback when content is submitted for ingestion */
  onSubmit: (input: { type: SourceType; value: string | File }) => Promise<void>;
  /** Whether the console is disabled */
  disabled?: boolean;
  /** Custom class name */
  className?: string;
}

/** Detect the type of URL */
function detectSourceType(url: string): SourceType | null {
  const urlLower = url.toLowerCase();
  
  // YouTube detection
  if (
    urlLower.includes('youtube.com') ||
    urlLower.includes('youtu.be')
  ) {
    return 'youtube';
  }
  
  // PDF detection
  if (urlLower.endsWith('.pdf')) {
    return 'pdf';
  }
  
  // Web URL
  if (urlLower.startsWith('http://') || urlLower.startsWith('https://')) {
    return 'web';
  }
  
  return null;
}

/** Get icon for source type */
function getSourceIcon(type: SourceType) {
  switch (type) {
    case 'youtube':
      return <Youtube className="w-4 h-4" />;
    case 'pdf':
      return <FileText className="w-4 h-4" />;
    case 'web':
      return <Globe className="w-4 h-4" />;
    case 'file':
      return <Upload className="w-4 h-4" />;
  }
}

/**
 * InputConsole - Unified input for sharing content with the Terminator
 * 
 * Supports:
 * - URL input (web articles, YouTube, PDFs)
 * - File drag & drop (PDF, DOCX, TXT)
 * - Validation and type detection
 */
export const InputConsole = ({
  onSubmit,
  disabled = false,
  className,
}: InputConsoleProps) => {
  const [urlInput, setUrlInput] = useState('');
  const [status, setStatus] = useState<UploadStatus>('idle');
  const [statusMessage, setStatusMessage] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [detectedType, setDetectedType] = useState<SourceType | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Detect URL type as user types
  const handleUrlChange = (e: ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setUrlInput(value);
    setDetectedType(value.trim() ? detectSourceType(value) : null);
  };

  // Handle URL submission
  const handleUrlSubmit = useCallback(async () => {
    const url = urlInput.trim();
    if (!url) return;

    const type = detectSourceType(url);
    if (!type) {
      setStatus('error');
      setStatusMessage('Invalid URL format');
      return;
    }

    setStatus('uploading');
    setStatusMessage(`Processing ${type === 'youtube' ? 'video' : type}...`);

    try {
      await onSubmit({ type, value: url });
      setStatus('success');
      setStatusMessage('Content shared with T-800');
      setUrlInput('');
      setDetectedType(null);
      
      // Reset after delay
      setTimeout(() => {
        setStatus('idle');
        setStatusMessage('');
      }, 3000);
    } catch (error) {
      setStatus('error');
      setStatusMessage(error instanceof Error ? error.message : 'Failed to process');
    }
  }, [urlInput, onSubmit]);

  // Handle file drop
  const handleDrop = useCallback(
    async (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);

      const file = e.dataTransfer.files[0];
      if (!file) return;

      // Validate file type
      const validTypes = ['application/pdf', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
      if (!validTypes.includes(file.type)) {
        setStatus('error');
        setStatusMessage('Unsupported file type');
        return;
      }

      setStatus('uploading');
      setStatusMessage(`Uploading ${file.name}...`);

      try {
        await onSubmit({ type: 'file', value: file });
        setStatus('success');
        setStatusMessage(`${file.name} shared with T-800`);
        
        setTimeout(() => {
          setStatus('idle');
          setStatusMessage('');
        }, 3000);
      } catch (error) {
        setStatus('error');
        setStatusMessage(error instanceof Error ? error.message : 'Failed to upload');
      }
    },
    [onSubmit]
  );

  // Handle file input change
  const handleFileChange = useCallback(
    async (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      setStatus('uploading');
      setStatusMessage(`Uploading ${file.name}...`);

      try {
        await onSubmit({ type: 'file', value: file });
        setStatus('success');
        setStatusMessage(`${file.name} shared with T-800`);
        
        setTimeout(() => {
          setStatus('idle');
          setStatusMessage('');
        }, 3000);
      } catch (error) {
        setStatus('error');
        setStatusMessage(error instanceof Error ? error.message : 'Failed to upload');
      }
      
      // Reset input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    },
    [onSubmit]
  );

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && urlInput.trim()) {
      e.preventDefault();
      handleUrlSubmit();
    }
  };

  return (
    <div
      className={cn(
        'hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-4',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-4 pb-2 border-b border-terminator-border">
        <span className="text-terminator-red text-xs font-mono tracking-wider">
          ▸ SHARE WITH TERMINATOR
        </span>
      </div>

      {/* Drop Zone */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={cn(
          'border-2 border-dashed rounded-lg p-4 mb-3 transition-all duration-200',
          'flex flex-col items-center justify-center min-h-[80px]',
          isDragging
            ? 'border-terminator-cyan bg-terminator-cyan/10'
            : 'border-terminator-border hover:border-terminator-red/50',
          disabled && 'opacity-50 pointer-events-none'
        )}
      >
        <Upload className={cn(
          'w-6 h-6 mb-2 transition-colors',
          isDragging ? 'text-terminator-cyan' : 'text-terminator-text-dim'
        )} />
        <p className="text-xs font-mono text-terminator-text-dim text-center">
          {isDragging ? 'DROP FILE HERE' : 'Drop PDF, DOCX, or TXT'}
        </p>
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled || status === 'uploading'}
          className="mt-2 text-xs font-mono text-terminator-cyan hover:text-terminator-cyan/80 transition-colors"
        >
          or click to browse
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>

      {/* URL Input */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-terminator-text-dim">
              {detectedType ? getSourceIcon(detectedType) : <Link className="w-4 h-4" />}
            </div>
            <input
              type="text"
              value={urlInput}
              onChange={handleUrlChange}
              onKeyDown={handleKeyDown}
              placeholder="Paste URL (YouTube, article, PDF)..."
              disabled={disabled || status === 'uploading'}
              className={cn(
                'w-full pl-10 pr-10 py-2.5 rounded',
                'bg-terminator-darker border border-terminator-border',
                'text-sm font-mono text-terminator-text placeholder:text-terminator-text-dim',
                'focus:outline-none focus:border-terminator-red focus:shadow-glow-red',
                'transition-all duration-200',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
            />
            {urlInput && (
              <button
                onClick={() => {
                  setUrlInput('');
                  setDetectedType(null);
                }}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-terminator-text-dim hover:text-terminator-text"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
          <button
            onClick={handleUrlSubmit}
            disabled={!urlInput.trim() || disabled || status === 'uploading'}
            className={cn(
              'btn-hud-primary px-4 py-2.5 flex items-center gap-2',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
          >
            {status === 'uploading' ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Upload className="w-4 h-4" />
            )}
            <span className="hidden sm:inline">Share</span>
          </button>
        </div>

        {/* Status Message */}
        {statusMessage && (
          <div
            className={cn(
              'flex items-center gap-2 text-xs font-mono p-2 rounded',
              status === 'success' && 'bg-green-500/10 text-green-400',
              status === 'error' && 'bg-terminator-red/10 text-terminator-red',
              status === 'uploading' && 'bg-terminator-cyan/10 text-terminator-cyan'
            )}
          >
            {status === 'success' && <CheckCircle className="w-4 h-4" />}
            {status === 'error' && <AlertCircle className="w-4 h-4" />}
            {status === 'uploading' && <Loader2 className="w-4 h-4 animate-spin" />}
            <span>{statusMessage}</span>
          </div>
        )}
      </div>

      {/* Supported formats */}
      <div className="mt-3 pt-3 border-t border-terminator-border">
        <p className="text-[10px] font-mono text-terminator-text-dim">
          SUPPORTED: YouTube • Web articles • PDF • DOCX • TXT
        </p>
      </div>
    </div>
  );
};

