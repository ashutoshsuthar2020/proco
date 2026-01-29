import React, { useState } from 'react';
import { apiService, SearchResult } from '../services/api';
import { LoadingSpinner } from './Loading';

interface SearchBarProps {
    onResults?: (results: SearchResult[]) => void;
}

export const SearchBar: React.FC<SearchBarProps> = ({ onResults }) => {
    const [query, setQuery] = useState('');
    const [isSearching, setIsSearching] = useState(false);
    const [results, setResults] = useState<SearchResult[]>([]);
    const [error, setError] = useState<string | null>(null);

    const handleSearch = async () => {
        if (!query.trim()) return;

        setIsSearching(true);
        setError(null);

        try {
            const searchResults = await apiService.searchDocuments(query.trim(), 10, 0.3);
            setResults(searchResults);
            onResults?.(searchResults);
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Search failed. Please try again.');
            console.error('Search error:', err);
        } finally {
            setIsSearching(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    };

    return (
        <div className="w-full">
            <div className="relative">
                <div className="relative flex items-center">
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Search documents by content, keywords, or topics..."
                        className="w-full px-4 py-3 pr-12 text-gray-900 placeholder-gray-500 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                        disabled={isSearching}
                    />

                    <button
                        onClick={handleSearch}
                        disabled={isSearching || !query.trim()}
                        className="absolute right-2 p-2 text-gray-400 hover:text-primary-600 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isSearching ? (
                            <LoadingSpinner size="sm" />
                        ) : (
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                            </svg>
                        )}
                    </button>
                </div>
            </div>

            {error && (
                <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-md">
                    <p className="text-sm text-red-700">{error}</p>
                </div>
            )}

            {results.length > 0 && (
                <div className="mt-4">
                    <h3 className="text-sm font-medium text-gray-900 mb-3">
                        Search Results ({results.length})
                    </h3>
                    <div className="space-y-3">
                        {results.map((result, index) => (
                            <SearchResultCard key={`${result.document_id}-${index}`} result={result} />
                        ))}
                    </div>
                </div>
            )}

            {query && !isSearching && results.length === 0 && !error && (
                <div className="mt-4 text-center py-8">
                    <div className="text-gray-500">
                        <svg className="w-12 h-12 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        <p className="text-sm">No similar documents found for "{query}"</p>
                        <p className="text-xs text-gray-400 mt-1">Try different keywords or upload more documents</p>
                    </div>
                </div>
            )}
        </div>
    );
};

interface SearchResultCardProps {
    result: SearchResult;
}

const SearchResultCard: React.FC<SearchResultCardProps> = ({ result }) => {
    const similarityPercentage = Math.round(result.similarity_score * 100);

    const getSimilarityColor = (score: number) => {
        if (score >= 0.8) return 'text-green-600 bg-green-100';
        if (score >= 0.6) return 'text-yellow-600 bg-yellow-100';
        return 'text-blue-600 bg-blue-100';
    };

    return (
        <div className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between mb-3">
                <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2 mb-1">
                        <h4 className="text-sm font-medium text-gray-900 truncate" title={result.filename}>
                            {result.filename}
                        </h4>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getSimilarityColor(result.similarity_score)}`}>
                            {similarityPercentage}% match
                        </span>
                    </div>
                    <p className="text-xs text-gray-500">
                        Document ID: {result.document_id.split('-')[0]}...
                    </p>
                </div>
            </div>

            <div className="text-sm text-gray-800 mb-3">
                <p className="line-clamp-3">
                    {result.text_preview}
                </p>
            </div>

            {result.metadata && Object.keys(result.metadata).length > 0 && (
                <div className="border-t border-gray-100 pt-3">
                    <details className="text-xs text-gray-600">
                        <summary className="cursor-pointer hover:text-gray-800 select-none">
                            Document Metadata
                        </summary>
                        <div className="mt-2 bg-gray-50 p-2 rounded">
                            <pre className="whitespace-pre-wrap text-gray-700">
                                {JSON.stringify(result.metadata, null, 2)}
                            </pre>
                        </div>
                    </details>
                </div>
            )}
        </div>
    );
};