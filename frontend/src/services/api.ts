import axios from 'axios';

// Always use relative URLs in production Docker container
// For local development, can be overridden with REACT_APP_API_URL
const API_BASE_URL = process.env.REACT_APP_API_URL || '';

export interface Document {
    id: string;
    filename: string;
    file_type: string;
    file_size: number;
    file_hash: string;
    created_at: string;
    updated_at?: string;
    doc_metadata?: Record<string, any>;
}

export interface Summary {
    id: string;
    document_id: string;
    summary_text: string;
    word_count: number;
    confidence_score?: number;
    processing_time_seconds?: number;
    model_version?: string;
    summary_metadata?: Record<string, any>;
    created_at: string;
}

export interface UploadResponse {
    document_id: string;
    message: string;
    processing_status: string;
}

export interface HealthStats {
    status: string;
    timestamp: string;
    version: string;
    stats: {
        database: {
            total_documents: number;
            total_summaries: number;
            total_chunks: number;
            avg_summary_confidence: number;
        };
        cache: {
            status: string;
            used_memory?: string;
            connected_clients?: number;
        };
        vector_db: {
            status: string;
            collection_name: string;
            points_count: number;
            vectors_count: number;
        };
        system_health: string;
    };
}

export interface SearchResult {
    document_id: string;
    filename: string;
    similarity_score: number;
    text_preview: string;
    metadata: Record<string, any>;
}

class ApiService {
    private api = axios.create({
        baseURL: API_BASE_URL,
        timeout: 30000,
    });

    // Helper to get correct API path
    private getApiPath(path: string): string {
        // Always use /api prefix for nginx proxy in Docker container
        // For local dev, set REACT_APP_API_URL=http://localhost:8000 to bypass proxy
        return `/api${path}`;
    }

    async uploadDocument(
        file: File,
        extractImmediately: boolean = true,
        metadata?: Record<string, any>
    ): Promise<UploadResponse> {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('extract_immediately', extractImmediately.toString());

        if (metadata) {
            formData.append('doc_metadata', JSON.stringify(metadata));
        }

        const response = await this.api.post(this.getApiPath('/documents/ingest'), formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });

        return response.data;
    }

    async getDocument(documentId: string): Promise<Document> {
        const response = await this.api.get(this.getApiPath(`/documents/${documentId}`));
        return response.data;
    }

    async summarizeDocument(
        documentId: string,
        options: {
            force_regenerate?: boolean;
            summary_length?: string;
            focus_areas?: string[];
        } = {}
    ): Promise<Summary> {
        const requestBody = {
            document_id: documentId,
            ...options
        };
        const response = await this.api.post(this.getApiPath('/documents/summarize'), requestBody);
        return response.data;
    }

    async getSummary(documentId: string): Promise<Summary> {
        const response = await this.api.get(this.getApiPath(`/documents/${documentId}/summary`));
        return response.data;
    }

    async searchDocuments(
        query: string,
        limit: number = 10,
        minSimilarity: number = 0.5
    ): Promise<SearchResult[]> {
        const response = await this.api.post(this.getApiPath('/search/semantic'), {
            query,
            limit,
            min_similarity: minSimilarity,
        });
        return response.data.results || [];
    }

    async getHealthStats(): Promise<HealthStats> {
        const response = await this.api.get(this.getApiPath('/health'));
        return response.data;
    }

    async listDocuments(page: number = 1, pageSize: number = 20): Promise<Document[]> {
        // This endpoint might need to be implemented in the backend
        try {
            const response = await this.api.get(this.getApiPath(`/documents?page=${page}&page_size=${pageSize}`));
            return response.data.documents || [];
        } catch (error) {
            // Fallback: return empty array if endpoint doesn't exist
            return [];
        }
    }
}

export const apiService = new ApiService();