import { useState, useCallback } from 'react';
import type { DocumentMetadata } from '../types';

type SourceType = 'pdf' | 'youtube' | 'web' | 'file';

interface IngestInput {
  type: SourceType;
  value: string | File;
}

interface UseDocumentsOptions {
  /** API endpoint for ingestion */
  ingestEndpoint?: string;
  /** API endpoint for listing documents */
  listEndpoint?: string;
}

interface UseDocumentsReturn {
  /** List of ingested documents */
  documents: DocumentMetadata[];
  /** Currently active/selected document */
  activeDocument: DocumentMetadata | null;
  /** Ingest new content */
  ingest: (input: IngestInput) => Promise<DocumentMetadata>;
  /** Refresh document list */
  refreshDocuments: () => Promise<void>;
  /** Set the active document */
  setActiveDocument: (doc: DocumentMetadata | null) => void;
  /** Whether currently ingesting */
  isIngesting: boolean;
  /** Whether loading documents */
  isLoading: boolean;
  /** Error message if any */
  error: string | null;
}

/**
 * useDocuments - Hook for managing document ingestion and listing
 * 
 * Handles:
 * - URL ingestion (YouTube, web articles, PDFs)
 * - File upload
 * - Document listing
 * - Active document tracking
 */
export function useDocuments(options: UseDocumentsOptions = {}): UseDocumentsReturn {
  const {
    ingestEndpoint = '/api/ingest',
    listEndpoint = '/api/documents',
  } = options;

  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [activeDocument, setActiveDocument] = useState<DocumentMetadata | null>(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Ingest URL content
  const ingestUrl = async (type: SourceType, url: string): Promise<DocumentMetadata> => {
    const response = await fetch(ingestEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ type, url }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `Ingestion failed: HTTP ${response.status}`);
    }

    return response.json();
  };

  // Upload file content
  const ingestFile = async (file: File): Promise<DocumentMetadata> => {
    const formData = new FormData();
    formData.append('file', file);

    // Use specific endpoint for file uploads
    const fileEndpoint = ingestEndpoint.endsWith('/') 
      ? `${ingestEndpoint}file` 
      : `${ingestEndpoint}/file`;

    const response = await fetch(fileEndpoint, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `Upload failed: HTTP ${response.status}`);
    }

    return response.json();
  };

  // Main ingest function
  const ingest = useCallback(async (input: IngestInput): Promise<DocumentMetadata> => {
    setIsIngesting(true);
    setError(null);

    try {
      let result: DocumentMetadata;

      if (input.type === 'file' && input.value instanceof File) {
        result = await ingestFile(input.value);
      } else if (typeof input.value === 'string') {
        result = await ingestUrl(input.type, input.value);
      } else {
        throw new Error('Invalid input');
      }

      // Add to documents list
      setDocuments((prev) => [result, ...prev]);
      
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Ingestion failed';
      setError(errorMessage);
      throw err;
    } finally {
      setIsIngesting(false);
    }
  }, [ingestEndpoint]);

  // Refresh document list
  const refreshDocuments = useCallback(async (): Promise<void> => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(listEndpoint);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch documents: HTTP ${response.status}`);
      }

      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load documents';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [listEndpoint]);

  return {
    documents,
    activeDocument,
    ingest,
    refreshDocuments,
    setActiveDocument,
    isIngesting,
    isLoading,
    error,
  };
}

