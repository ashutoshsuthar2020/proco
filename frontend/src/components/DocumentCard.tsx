import React from 'react';
import { Document } from '../services/api';

interface DocumentCardProps {
    document: Document;
    onSummarize?: () => void;
    onDelete?: () => void;
    hasSummary?: boolean;
    isProcessing?: boolean;
}

export const DocumentCard: React.FC<DocumentCardProps> = ({
    document,
    onSummarize,
    onDelete,
    hasSummary = false,
    isProcessing = false
}) => {
    const formatFileSize = (bytes: number) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    const getFileIcon = (fileType: string) => {
        switch (fileType.toLowerCase()) {
            case 'pdf':
                return '📄';
            case 'docx':
                return '📝';
            case 'txt':
                return '📃';
            default:
                return '📄';
        }
    };

    return (
        <div className="bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition-all duration-200">
            <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3 flex-1 min-w-0">
                        <div className="text-2xl">
                            {getFileIcon(document.file_type)}
                        </div>
                        <div className="flex-1 min-w-0">
                            <h3 className="text-sm font-semibold text-gray-900 truncate" title={document.filename}>
                                {document.filename}
                            </h3>
                            <div className="flex items-center space-x-3 mt-1 text-xs text-gray-500">
                                <span>{document.file_type.toUpperCase()}</span>
                                <span>{formatFileSize(document.file_size)}</span>
                                <span>{formatDate(document.created_at)}</span>
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center space-x-2">
                        {hasSummary && (
                            <div className="flex items-center space-x-1 text-xs text-green-600">
                                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                                <span>Summarized</span>
                            </div>
                        )}

                        {isProcessing && (
                            <div className="flex items-center space-x-1 text-xs text-blue-600">
                                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                                <span>Processing</span>
                            </div>
                        )}
                    </div>
                </div>

                {document.doc_metadata && Object.keys(document.doc_metadata).length > 0 && (
                    <div className="mb-4">
                        <details className="text-xs text-gray-600">
                            <summary className="cursor-pointer hover:text-gray-800 select-none">
                                Document Metadata
                            </summary>
                            <div className="mt-2 bg-gray-50 p-2 rounded text-xs">
                                <pre className="whitespace-pre-wrap text-gray-700">
                                    {JSON.stringify(document.doc_metadata, null, 2)}
                                </pre>
                            </div>
                        </details>
                    </div>
                )}

                <div className="flex items-center justify-between">
                    <div className="text-xs text-gray-500">
                        ID: {document.id.split('-')[0]}...
                    </div>

                    <div className="flex space-x-2">
                        {onSummarize && (
                            <button
                                onClick={onSummarize}
                                disabled={isProcessing}
                                className={`
                  px-3 py-1.5 text-xs font-medium rounded-md transition-colors
                  ${isProcessing
                                        ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                                        : hasSummary
                                            ? 'bg-blue-50 text-blue-700 hover:bg-blue-100'
                                            : 'bg-primary-50 text-primary-700 hover:bg-primary-100'
                                    }
                `}
                            >
                                {isProcessing ? 'Processing...' : hasSummary ? 'View Summary' : 'Summarize'}
                            </button>
                        )}

                        {onDelete && (
                            <button
                                onClick={onDelete}
                                className="px-3 py-1.5 text-xs font-medium text-red-700 bg-red-50 rounded-md hover:bg-red-100 transition-colors"
                            >
                                Delete
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

interface DocumentListProps {
    documents: Document[];
    onSummarize?: (document: Document) => void;
    onDelete?: (document: Document) => void;
    loading?: boolean;
}

export const DocumentList: React.FC<DocumentListProps> = ({
    documents,
    onSummarize,
    onDelete,
    loading = false
}) => {
    if (loading) {
        return (
            <div className="space-y-4">
                {[1, 2, 3].map((i) => (
                    <div key={i} className="bg-white border border-gray-200 rounded-lg p-6">
                        <div className="animate-pulse">
                            <div className="flex items-center space-x-3 mb-4">
                                <div className="w-6 h-6 bg-gray-300 rounded"></div>
                                <div className="space-y-2 flex-1">
                                    <div className="h-4 bg-gray-300 rounded w-3/4"></div>
                                    <div className="h-3 bg-gray-300 rounded w-1/2"></div>
                                </div>
                            </div>
                            <div className="flex justify-between">
                                <div className="h-3 bg-gray-300 rounded w-20"></div>
                                <div className="h-6 bg-gray-300 rounded w-24"></div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        );
    }

    if (documents.length === 0) {
        return (
            <div className="text-center py-12">
                <div className="text-4xl mb-4">📄</div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">No documents yet</h3>
                <p className="text-gray-600">Upload your first document to get started with AI summarization.</p>
            </div>
        );
    }

    return (
        <div className="space-y-4">
            {documents.map((document) => (
                <DocumentCard
                    key={document.id}
                    document={document}
                    onSummarize={() => onSummarize?.(document)}
                    onDelete={() => onDelete?.(document)}
                />
            ))}
        </div>
    );
};