import React, { useState, useEffect } from 'react';
import { apiService, Document, Summary, UploadResponse } from '../services/api';
import { FileUpload } from '../components/FileUpload';
import { ProgressBar, LoadingSpinner } from '../components/Loading';
import { SummaryDisplay } from '../components/SummaryDisplay';
import { DocumentCard, DocumentList } from '../components/DocumentCard';
import { SearchBar } from '../components/SearchBar';

type AppState = 'upload' | 'processing' | 'summary' | 'search';

interface UploadState {
    file: File | null;
    isUploading: boolean;
    uploadProgress: number;
    uploadResponse: UploadResponse | null;
    error: string | null;
}

interface ProcessingState {
    document: Document | null;
    isGeneratingSummary: boolean;
    summary: Summary | null;
    error: string | null;
}

export const DocumentSummarizer: React.FC = () => {
    const [currentState, setCurrentState] = useState<AppState>('upload');

    const [uploadState, setUploadState] = useState<UploadState>({
        file: null,
        isUploading: false,
        uploadProgress: 0,
        uploadResponse: null,
        error: null,
    });

    const [processingState, setProcessingState] = useState<ProcessingState>({
        document: null,
        isGeneratingSummary: false,
        summary: null,
        error: null,
    });

    const [recentDocuments] = useState<Document[]>([]);
    const [systemHealth, setSystemHealth] = useState<string>('checking');

    // Check system health on mount
    useEffect(() => {
        checkSystemHealth();
        // loadRecentDocuments();
    }, []);

    const checkSystemHealth = async () => {
        try {
            const health = await apiService.getHealthStats();
            setSystemHealth(health.stats.system_health);
        } catch (error) {
            setSystemHealth('error');
            console.error('Health check failed:', error);
        }
    };

    const resetUploadState = () => {
        setUploadState({
            file: null,
            isUploading: false,
            uploadProgress: 0,
            uploadResponse: null,
            error: null,
        });
    };

    const resetProcessingState = () => {
        setProcessingState({
            document: null,
            isGeneratingSummary: false,
            summary: null,
            error: null,
        });
    };

    const handleFileSelect = async (file: File) => {
        setUploadState(prev => ({
            ...prev,
            file,
            error: null,
            uploadProgress: 0
        }));

        // Auto-upload immediately
        await handleUpload(file);
    };

    const handleUpload = async (file: File) => {
        setUploadState(prev => ({ ...prev, isUploading: true, error: null }));
        setCurrentState('processing');

        try {
            // Simulate upload progress
            const progressInterval = setInterval(() => {
                setUploadState(prev => ({
                    ...prev,
                    uploadProgress: Math.min(prev.uploadProgress + 10, 90)
                }));
            }, 200);

            const uploadResponse = await apiService.uploadDocument(file, true, {
                uploaded_from: 'web_interface',
                upload_timestamp: new Date().toISOString()
            });

            clearInterval(progressInterval);
            setUploadState(prev => ({
                ...prev,
                uploadProgress: 100,
                uploadResponse,
                isUploading: false
            }));

            // Fetch the uploaded document details
            const document = await apiService.getDocument(uploadResponse.document_id);
            setProcessingState(prev => ({ ...prev, document }));

            // Auto-generate summary
            setTimeout(() => {
                handleGenerateSummary(uploadResponse.document_id);
            }, 1000);

        } catch (error: any) {
            setUploadState(prev => ({
                ...prev,
                isUploading: false,
                uploadProgress: 0,
                error: error.response?.data?.detail || 'Upload failed. Please try again.'
            }));
            setCurrentState('upload');
        }
    };

    const handleGenerateSummary = async (documentId: string) => {
        setProcessingState(prev => ({ ...prev, isGeneratingSummary: true, error: null }));

        try {
            const summary = await apiService.summarizeDocument(documentId, {
                summary_length: 'medium',
                force_regenerate: false
            });

            setProcessingState(prev => ({ ...prev, summary, isGeneratingSummary: false }));
            setCurrentState('summary');
        } catch (error: any) {
            setProcessingState(prev => ({
                ...prev,
                isGeneratingSummary: false,
                error: error.response?.data?.detail || 'Summary generation failed. Please try again.'
            }));
        }
    };

    const handleStartOver = () => {
        resetUploadState();
        resetProcessingState();
        setCurrentState('upload');
    };

    const handleShowSearch = () => {
        setCurrentState('search');
    };

    const renderSystemStatus = () => (
        <div className="mb-6 p-4 bg-white border border-gray-200 rounded-lg">
            <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                    <div className={`w-3 h-3 rounded-full ${systemHealth === 'healthy' ? 'bg-green-400' :
                        systemHealth === 'checking' ? 'bg-yellow-400' : 'bg-red-400'
                        }`}></div>
                    <span className="text-sm font-medium text-gray-700">
                        System Status: {systemHealth === 'healthy' ? 'All Systems Operational' :
                            systemHealth === 'checking' ? 'Checking...' : 'Service Unavailable'}
                    </span>
                </div>

                <div className="flex space-x-2">
                    <button
                        onClick={handleShowSearch}
                        className="px-3 py-1 text-xs font-medium text-blue-700 bg-blue-50 rounded-md hover:bg-blue-100 transition-colors"
                    >
                        🔍 Search Documents
                    </button>

                    {currentState !== 'upload' && (
                        <button
                            onClick={handleStartOver}
                            className="px-3 py-1 text-xs font-medium text-gray-700 bg-gray-50 rounded-md hover:bg-gray-100 transition-colors"
                        >
                            ↻ New Document
                        </button>
                    )}
                </div>
            </div>
        </div>
    );

    const renderUploadSection = () => (
        <div className="max-w-2xl mx-auto">
            <div className="text-center mb-8">
                <h1 className="text-3xl font-bold text-gray-900 mb-4">
                    🤖 AI Document Summarizer
                </h1>
                <p className="text-lg text-gray-600">
                    Upload your documents and get intelligent summaries powered by multi-agent AI
                </p>
                <div className="mt-4 flex justify-center space-x-4 text-sm text-gray-500">
                    <span className="flex items-center">
                        <div className="w-2 h-2 bg-blue-400 rounded-full mr-2"></div>
                        PDF Support
                    </span>
                    <span className="flex items-center">
                        <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
                        Word Documents
                    </span>
                    <span className="flex items-center">
                        <div className="w-2 h-2 bg-purple-400 rounded-full mr-2"></div>
                        Text Files
                    </span>
                </div>
            </div>

            <FileUpload
                onFileSelect={handleFileSelect}
                isUploading={uploadState.isUploading}
            />

            {uploadState.error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
                    <div className="flex items-center">
                        <svg className="w-5 h-5 text-red-400 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <p className="text-sm text-red-700">{uploadState.error}</p>
                    </div>
                </div>
            )}
        </div>
    );

    const renderProcessingSection = () => (
        <div className="max-w-2xl mx-auto">
            <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Processing Document</h2>
                <p className="text-gray-600">
                    {uploadState.file?.name} • {uploadState.file ? (uploadState.file.size / 1024 / 1024).toFixed(1) : 0}MB
                </p>
            </div>

            <div className="space-y-6">
                <div className="bg-white border border-gray-200 rounded-lg p-6">
                    <ProgressBar
                        progress={uploadState.uploadProgress}
                        status={uploadState.isUploading ? 'Uploading document...' : 'Upload complete'}
                        color={uploadState.isUploading ? 'blue' : 'green'}
                    />
                </div>

                {processingState.document && (
                    <DocumentCard
                        document={processingState.document}
                        isProcessing={processingState.isGeneratingSummary}
                    />
                )}

                {processingState.isGeneratingSummary && (
                    <div className="bg-white border border-gray-200 rounded-lg p-6">
                        <div className="flex items-center justify-center space-x-3">
                            <LoadingSpinner size="md" />
                            <div className="text-gray-700">
                                <p className="font-medium">Generating AI Summary</p>
                                <p className="text-sm text-gray-500">Multi-agent processing in progress...</p>
                            </div>
                        </div>
                    </div>
                )}

                {processingState.error && (
                    <div className="p-4 bg-red-50 border border-red-200 rounded-md">
                        <p className="text-sm text-red-700">{processingState.error}</p>
                    </div>
                )}
            </div>
        </div>
    );

    const renderSummarySection = () => (
        <div className="max-w-4xl mx-auto">
            <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Summary Generated</h2>
                <p className="text-gray-600">
                    {processingState.document?.filename} successfully processed
                </p>
            </div>

            {processingState.summary && (
                <SummaryDisplay
                    summary={processingState.summary}
                    isGenerating={processingState.isGeneratingSummary}
                />
            )}

            <div className="mt-8 flex justify-center space-x-4">
                <button
                    onClick={handleStartOver}
                    className="px-6 py-2 bg-primary-600 text-white font-medium rounded-lg hover:bg-primary-700 transition-colors"
                >
                    Upload Another Document
                </button>

                <button
                    onClick={handleShowSearch}
                    className="px-6 py-2 bg-gray-100 text-gray-700 font-medium rounded-lg hover:bg-gray-200 transition-colors"
                >
                    Search Similar Documents
                </button>
            </div>
        </div>
    );

    const renderSearchSection = () => (
        <div className="max-w-4xl mx-auto">
            <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Search Documents</h2>
                <p className="text-gray-600">
                    Find similar documents using semantic search
                </p>
            </div>

            <div className="mb-8">
                <SearchBar />
            </div>

            {recentDocuments.length > 0 && (
                <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Documents</h3>
                    <DocumentList documents={recentDocuments} />
                </div>
            )}
        </div>
    );

    return (
        <div className="min-h-screen bg-gray-50">
            <div className="container mx-auto px-4 py-8">
                {renderSystemStatus()}

                {currentState === 'upload' && renderUploadSection()}
                {currentState === 'processing' && renderProcessingSection()}
                {currentState === 'summary' && renderSummarySection()}
                {currentState === 'search' && renderSearchSection()}
            </div>
        </div>
    );
};