import React from 'react';
import { Summary } from '../services/api';

interface SummaryDisplayProps {
    summary: Summary;
    isGenerating?: boolean;
}

export const SummaryDisplay: React.FC<SummaryDisplayProps> = ({
    summary,
    isGenerating = false
}) => {
    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    const formatStructuredText = (text: string) => {
        // First, convert escaped newlines to actual newlines
        let formatted = text
            .replace(/\\n/g, '\n')
            .replace(/\\t/g, '\t');

        // Split into lines and process each line
        const lines = formatted.split('\n');
        const processedLines: string[] = [];

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) {
                processedLines.push('<br>');
                continue;
            }

            // Handle emoji headers (e.g., "📋 **Key Points:**")
            if (/^([\u{1F300}-\u{1F9FF}]*)\s*\*\*(.*?)\*\*:?\s*$/u.test(line)) {
                const match = line.match(/^([\u{1F300}-\u{1F9FF}]*)\s*\*\*(.*?)\*\*:?\s*$/u);
                if (match) {
                    processedLines.push(
                        `<h3 class="text-lg font-semibold text-gray-900 mb-3 mt-6 flex items-center"><span class="text-xl mr-2">${match[1]}</span>${match[2]}</h3>`
                    );
                    continue;
                }
            }

            // Handle bullet points
            if (/^\s*[•\-]\s+(.+)$/.test(line)) {
                const match = line.match(/^\s*[•\-]\s+(.+)$/);
                if (match) {
                    processedLines.push(`<li class="mb-2 text-gray-700 list-disc ml-6">${match[1]}</li>`);
                    continue;
                }
            }

            // Handle warning/info notes
            if (/^([⚠️🔔📝])\s*\*\*(.*?)\*\*(.+)$/.test(line)) {
                const match = line.match(/^([⚠️🔔📝])\s*\*\*(.*?)\*\*(.+)$/);
                if (match) {
                    processedLines.push(
                        `<div class="bg-yellow-50 border-l-4 border-yellow-400 p-3 my-3 rounded-r"><div class="flex items-center"><span class="text-lg mr-2">${match[1]}</span><strong class="text-yellow-800">${match[2]}</strong></div><p class="text-yellow-700 mt-1">${match[3]}</p></div>`
                    );
                    continue;
                }
            }

            // Handle regular text with bold formatting
            const boldFormatted = line.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-gray-800">$1</strong>');
            processedLines.push(`<p class="text-gray-700 mb-3 leading-relaxed">${boldFormatted}</p>`);
        }

        // Group consecutive list items
        const finalLines: string[] = [];
        let inList = false;

        for (const line of processedLines) {
            if (line.startsWith('<li')) {
                if (!inList) {
                    finalLines.push('<ul class="space-y-1 mb-4">');
                    inList = true;
                }
                finalLines.push(line);
            } else {
                if (inList) {
                    finalLines.push('</ul>');
                    inList = false;
                }
                finalLines.push(line);
            }
        }

        if (inList) {
            finalLines.push('</ul>');
        }

        return finalLines.join('');
    };

    const getConfidenceColor = (score?: number) => {
        if (!score) return 'text-gray-500';
        if (score >= 0.8) return 'text-green-600';
        if (score >= 0.6) return 'text-yellow-600';
        return 'text-red-600';
    };

    const getConfidenceBg = (score?: number) => {
        if (!score) return 'bg-gray-100';
        if (score >= 0.8) return 'bg-green-100';
        if (score >= 0.6) return 'bg-yellow-100';
        return 'bg-red-100';
    };

    return (
        <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
            <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900">
                        📄 Document Summary
                    </h3>

                    <div className="flex items-center space-x-4 text-sm text-gray-500">
                        {summary.confidence_score && (
                            <div className="flex items-center">
                                <span className="mr-1">Confidence:</span>
                                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getConfidenceBg(summary.confidence_score)} ${getConfidenceColor(summary.confidence_score)}`}>
                                    {Math.round(summary.confidence_score * 100)}%
                                </span>
                            </div>
                        )}

                        <div>
                            {summary.word_count} words
                        </div>

                        {summary.processing_time_seconds && (
                            <div>
                                Generated in {summary.processing_time_seconds.toFixed(1)}s
                            </div>
                        )}
                    </div>
                </div>

                {isGenerating ? (
                    <div className="flex items-center justify-center py-8">
                        <div className="animate-spin-slow w-8 h-8 border-4 border-primary-200 border-t-primary-500 rounded-full"></div>
                        <span className="ml-3 text-gray-600">Generating summary...</span>
                    </div>
                ) : (
                    <div className="prose max-w-none">
                        <div
                            className="structured-summary"
                            dangerouslySetInnerHTML={{ __html: formatStructuredText(summary.summary_text) }}
                        />
                    </div>
                )}

                <div className="mt-6 pt-4 border-t border-gray-100">
                    <div className="flex justify-between items-center text-xs text-gray-500">
                        <div>
                            Created: {formatDate(summary.created_at)}
                        </div>

                        {summary.model_version && (
                            <div className="flex items-center">
                                <span className="w-2 h-2 bg-green-400 rounded-full mr-2"></span>
                                Model: {summary.model_version}
                            </div>
                        )}
                    </div>

                    {summary.summary_metadata && Object.keys(summary.summary_metadata).length > 0 && (
                        <details className="mt-3">
                            <summary className="cursor-pointer text-xs text-gray-600 hover:text-gray-800">
                                View Additional Details
                            </summary>
                            <div className="mt-2 text-xs bg-gray-50 p-3 rounded border">
                                <pre className="whitespace-pre-wrap text-gray-700">
                                    {JSON.stringify(summary.summary_metadata, null, 2)}
                                </pre>
                            </div>
                        </details>
                    )}
                </div>
            </div>
        </div>
    );
};

interface SummaryCardProps {
    summary: Summary;
    onViewFull?: () => void;
    compact?: boolean;
}

export const SummaryCard: React.FC<SummaryCardProps> = ({
    summary,
    onViewFull,
    compact = false
}) => {
    const truncateText = (text: string, maxLength: number) => {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength).trim() + '...';
    };

    const formatStructuredText = (text: string) => {
        // Simplified formatting for card view - handle escaped newlines
        let formatted = text
            .replace(/\\n/g, '\n')
            .replace(/\\t/g, ' ');

        const lines = formatted.split('\n');
        const processedLines: string[] = [];

        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;

            // Handle emoji headers
            if (/^([\u{1F300}-\u{1F9FF}]*)\s*\*\*(.*?)\*\*:?\s*$/u.test(trimmed)) {
                const match = trimmed.match(/^([\u{1F300}-\u{1F9FF}]*)\s*\*\*(.*?)\*\*:?\s*$/u);
                if (match) {
                    processedLines.push(`<div class="font-semibold text-gray-900 mb-2 flex items-center"><span class="mr-1">${match[1]}</span>${match[2]}</div>`);
                    continue;
                }
            }

            // Handle bullet points  
            if (/^\s*[•\-]\s+(.+)$/.test(trimmed)) {
                const match = trimmed.match(/^\s*[•\-]\s+(.+)$/);
                if (match) {
                    processedLines.push(`<div class="text-sm text-gray-700 ml-2 mb-1">• ${match[1]}</div>`);
                    continue;
                }
            }

            // Handle regular text
            const boldFormatted = trimmed.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>');
            processedLines.push(`<div class="text-gray-700 mb-1">${boldFormatted}</div>`);
        }

        return processedLines.join('');
    };

    return (
        <div className={`bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition-shadow ${compact ? 'p-4' : 'p-6'}`}>
            <div className="space-y-3">
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                        <span className="text-sm font-medium text-gray-700">Summary Ready</span>
                    </div>

                    {summary.confidence_score && (
                        <span className="text-xs text-gray-500">
                            {Math.round(summary.confidence_score * 100)}% confidence
                        </span>
                    )}
                </div>

                <div className={`structured-summary-card ${compact ? 'text-sm' : ''}`}>
                    {compact ? (
                        <p className="text-gray-800">
                            {truncateText(summary.summary_text.replace(/[\u{1F300}-\u{1F9FF}]|\*\*/gu, ''), 150)}
                        </p>
                    ) : (
                        <div dangerouslySetInnerHTML={{ __html: formatStructuredText(summary.summary_text) }} />
                    )}
                </div>

                <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>{summary.word_count} words</span>

                    {onViewFull && compact && (
                        <button
                            onClick={onViewFull}
                            className="text-primary-600 hover:text-primary-700 font-medium"
                        >
                            View Full Summary →
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};