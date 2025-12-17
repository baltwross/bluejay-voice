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
  File,
} from 'lucide-react';
import { cn } from '../utils';
import type { DocumentMetadata } from '../types';

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';
type SourceType = 'pdf' | 'youtube' | 'web' | 'file';

interface InputConsoleProps {
  /** Callback when content is submitted for ingestion */
  onSubmit: (input: { type: SourceType; value: string | File }) => Promise<void>;
  /** Whether the console is disabled */
  disabled?: boolean;
  /** Custom class name */
  className?: string;
  /** List of shared documents */
  documents?: DocumentMetadata[];
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
function getSourceIcon(type: string) {
  switch (type) {
    case 'youtube':
      return <Youtube className="w-4 h-4" />;
    case 'pdf':
      return <FileText className="w-4 h-4" />;
    case 'web':
      return <Globe className="w-4 h-4" />;
    case 'text':
    case 'file':
    default:
      return <File className="w-4 h-4" />;
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
  documents = [],
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
        // Compact padding on mobile
        'hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-2 sm:p-3',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-1.5 sm:gap-2 mb-2 sm:mb-3 pb-1.5 sm:pb-2 border-b border-terminator-border">
        <span className="text-terminator-red text-[10px] sm:text-xs font-mono tracking-wider">
          ▸ SHARE
        </span>
      </div>

      {/* Drop Zone - More compact on mobile */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={cn(
          'border-2 border-dashed rounded-lg p-1.5 sm:p-2 mb-1.5 sm:mb-2 transition-all duration-200',
          'flex flex-col items-center justify-center min-h-[44px] sm:min-h-[56px]',
          isDragging
            ? 'border-terminator-cyan bg-terminator-cyan/10'
            : 'border-terminator-border hover:border-terminator-red/50',
          disabled && 'opacity-50 pointer-events-none'
        )}
      >
        <Upload className={cn(
          'w-4 h-4 sm:w-5 sm:h-5 mb-0.5 sm:mb-1 transition-colors',
          isDragging ? 'text-terminator-cyan' : 'text-terminator-text-dim'
        )} />
        <p className="text-[10px] sm:text-xs font-mono text-terminator-text-dim text-center">
          {isDragging ? 'DROP FILE' : 'Drop or tap to upload'}
        </p>
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled || status === 'uploading'}
          className="mt-0.5 sm:mt-1 text-[10px] sm:text-xs font-mono text-terminator-cyan hover:text-terminator-cyan/80 transition-colors hidden sm:inline"
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
      <div className="space-y-1.5 sm:space-y-2">
        <div className="flex items-center gap-1.5 sm:gap-2">
          <div className="relative flex-1">
            <div className="absolute left-2 sm:left-3 top-1/2 -translate-y-1/2 text-terminator-text-dim">
              {detectedType ? (
                <div className="w-3.5 h-3.5 sm:w-4 sm:h-4">{getSourceIcon(detectedType)}</div>
              ) : (
                <Link className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
              )}
            </div>
            <input
              type="text"
              value={urlInput}
              onChange={handleUrlChange}
              onKeyDown={handleKeyDown}
              placeholder="Paste URL"
              disabled={disabled || status === 'uploading'}
              className={cn(
                'w-full pl-8 sm:pl-10 pr-8 sm:pr-10 py-1.5 sm:py-2 rounded',
                'bg-terminator-darker border border-terminator-border',
                'text-xs sm:text-sm font-mono text-terminator-text placeholder:text-terminator-text-dim',
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
                className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-terminator-text-dim hover:text-terminator-text"
              >
                <X className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
              </button>
            )}
          </div>
          <button
            onClick={handleUrlSubmit}
            disabled={!urlInput.trim() || disabled || status === 'uploading'}
            className={cn(
              'btn-hud-primary px-2 sm:px-4 py-1.5 sm:py-2 flex items-center gap-1 sm:gap-2',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
          >
            {status === 'uploading' ? (
              <Loader2 className="w-3.5 h-3.5 sm:w-4 sm:h-4 animate-spin" />
            ) : (
              <Upload className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
            )}
            <span className="hidden sm:inline text-xs sm:text-sm">Share</span>
          </button>
        </div>

        {/* Status Message */}
        {statusMessage && (
          <div
            className={cn(
              'flex items-center gap-1.5 sm:gap-2 text-[10px] sm:text-xs font-mono p-1.5 sm:p-2 rounded',
              status === 'success' && 'bg-green-500/10 text-green-400',
              status === 'error' && 'bg-terminator-red/10 text-terminator-red',
              status === 'uploading' && 'bg-terminator-cyan/10 text-terminator-cyan'
            )}
          >
            {status === 'success' && <CheckCircle className="w-3.5 h-3.5 sm:w-4 sm:h-4" />}
            {status === 'error' && <AlertCircle className="w-3.5 h-3.5 sm:w-4 sm:h-4" />}
            {status === 'uploading' && <Loader2 className="w-3.5 h-3.5 sm:w-4 sm:h-4 animate-spin" />}
            <span className="truncate">{statusMessage}</span>
          </div>
        )}
      </div>

      {/* Shared Files List - More compact on mobile */}
      {documents.length > 0 && (
        <div className="mt-2 sm:mt-3 pt-2 sm:pt-3 border-t border-terminator-border">
          <div className="flex items-center gap-1.5 sm:gap-2 mb-1 sm:mb-2">
            <span className="text-terminator-red text-[9px] sm:text-[10px] font-mono tracking-wider">
              ▸ SHARED [{documents.length}]
            </span>
          </div>
          
          <div className="relative group">
             {/* Visible List (Top 2) */}
             <div className="space-y-0.5 sm:space-y-1">
               {documents.slice(0, 2).map(doc => (
                 <div key={doc.id} className="flex items-center gap-1.5 sm:gap-2 text-[10px] sm:text-xs font-mono text-terminator-text-dim">
                   <div className="text-terminator-cyan/70 w-3.5 h-3.5 sm:w-4 sm:h-4 flex-shrink-0">
                     {getSourceIcon(doc.sourceType)}
                   </div>
                   <span className="truncate">{doc.title}</span>
                 </div>
               ))}
               
               {documents.length > 2 && (
                 <div className="flex items-center gap-1.5 sm:gap-2 text-[10px] sm:text-xs font-mono text-terminator-text-dim/70 pl-5 sm:pl-6 cursor-help hover:text-terminator-cyan transition-colors">
                   <span>+ {documents.length - 2} more...</span>
                 </div>
               )}
             </div>

             {/* Hover Drawer (All Files) - hidden on touch */}
             {documents.length > 2 && (
               <div className="absolute bottom-full left-0 w-full mb-2 hidden sm:group-hover:block z-50">
                 <div className="bg-terminator-darker/95 backdrop-blur-md border border-terminator-border rounded-lg shadow-2xl p-2 sm:p-3 max-h-[200px] overflow-y-auto custom-scrollbar">
                   <div className="text-[9px] sm:text-[10px] text-terminator-red font-mono mb-1.5 sm:mb-2 pb-1 border-b border-terminator-border/50 sticky top-0 bg-terminator-darker/95 backdrop-blur-md">
                     ALL FILES
                   </div>
                   <div className="space-y-1 sm:space-y-2">
                     {documents.map(doc => (
                       <div key={doc.id} className="flex items-center gap-1.5 sm:gap-2 text-[10px] sm:text-xs font-mono text-terminator-text hover:text-terminator-cyan transition-colors p-0.5 sm:p-1 rounded hover:bg-terminator-cyan/5">
                         <div className="shrink-0 text-terminator-cyan w-3.5 h-3.5 sm:w-4 sm:h-4">
                           {getSourceIcon(doc.sourceType)}
                         </div>
                         <span className="truncate" title={doc.title}>{doc.title}</span>
                       </div>
                     ))}
                   </div>
                 </div>
               </div>
             )}
          </div>
        </div>
      )}

      {/* Supported formats - hidden on very small screens */}
      <div className="hidden sm:block mt-1.5 sm:mt-2 pt-1.5 sm:pt-2 border-t border-terminator-border">
        <p className="text-[9px] sm:text-[10px] font-mono text-terminator-text-dim">
          SUPPORTED: YouTube • Web • PDF • DOCX • TXT
        </p>
      </div>
    </div>
  );
};



