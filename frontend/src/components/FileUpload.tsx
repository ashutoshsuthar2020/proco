import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';

interface FileUploadProps {
    onFileSelect: (file: File) => void;
    isUploading?: boolean;
    accept?: Record<string, string[]>;
    maxSize?: number;
}

export const FileUpload: React.FC<FileUploadProps> = ({
    onFileSelect,
    isUploading = false,
    accept = {
        'application/pdf': ['.pdf'],
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
        'text/plain': ['.txt'],
    },
    maxSize = 10 * 1024 * 1024, // 10MB
}) => {
    const [dragActive, setDragActive] = useState(false);

    const onDrop = useCallback(
        (acceptedFiles: File[]) => {
            if (acceptedFiles.length > 0) {
                onFileSelect(acceptedFiles[0]);
            }
            setDragActive(false);
        },
        [onFileSelect]
    );

    const onDragEnter = useCallback(() => {
        setDragActive(true);
    }, []);

    const onDragLeave = useCallback(() => {
        setDragActive(false);
    }, []);

    const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
        onDrop,
        onDragEnter,
        onDragLeave,
        accept,
        maxSize,
        multiple: false,
        disabled: isUploading,
    });

    const formatFileSize = (bytes: number) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    return (
        <div className="w-full">
            <div
                {...getRootProps()}
                className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200
          ${isDragActive || dragActive
                        ? 'border-primary-500 bg-primary-50'
                        : 'border-gray-300 hover:border-primary-400'
                    }
          ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
            >
                <input {...getInputProps()} />

                <div className="space-y-4">
                    <div className="flex justify-center">
                        <svg
                            className={`w-12 h-12 ${isDragActive || dragActive ? 'text-primary-500' : 'text-gray-400'
                                }`}
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                            />
                        </svg>
                    </div>

                    <div>
                        <p className="text-lg font-medium text-gray-900">
                            {isUploading
                                ? 'Uploading...'
                                : isDragActive || dragActive
                                    ? 'Drop your file here'
                                    : 'Upload a document'}
                        </p>
                        <p className="text-sm text-gray-600 mt-1">
                            {isUploading
                                ? 'Please wait while your document is being processed'
                                : 'Drag and drop or click to browse (PDF, DOCX, TXT)'}
                        </p>
                    </div>

                    <div className="text-xs text-gray-500">
                        Maximum file size: {formatFileSize(maxSize)}
                    </div>
                </div>
            </div>

            {fileRejections.length > 0 && (
                <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
                    <div className="text-sm text-red-700">
                        <strong>Upload Error:</strong>
                        {fileRejections.map(({ file, errors }) => (
                            <div key={file.name} className="mt-1">
                                {file.name}: {errors.map(e => e.message).join(', ')}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};