import React from 'react';

interface ProgressBarProps {
    progress: number;
    status?: string;
    showPercentage?: boolean;
    color?: 'blue' | 'green' | 'yellow' | 'red';
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
    progress,
    status,
    showPercentage = true,
    color = 'blue'
}) => {
    const colorClasses = {
        blue: 'bg-blue-500',
        green: 'bg-green-500',
        yellow: 'bg-yellow-500',
        red: 'bg-red-500'
    };

    const bgColorClasses = {
        blue: 'bg-blue-100',
        green: 'bg-green-100',
        yellow: 'bg-yellow-100',
        red: 'bg-red-100'
    };

    return (
        <div className="w-full">
            {status && (
                <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-gray-700">{status}</span>
                    {showPercentage && (
                        <span className="text-sm text-gray-500">{Math.round(progress)}%</span>
                    )}
                </div>
            )}

            <div className={`w-full ${bgColorClasses[color]} rounded-full h-2.5`}>
                <div
                    className={`${colorClasses[color]} h-2.5 rounded-full transition-all duration-300 ease-out`}
                    style={{ width: `${Math.min(progress, 100)}%` }}
                ></div>
            </div>
        </div>
    );
};

interface LoadingSpinnerProps {
    size?: 'sm' | 'md' | 'lg';
    color?: string;
    message?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
    size = 'md',
    color = 'text-primary-500',
    message
}) => {
    const sizeClasses = {
        sm: 'w-4 h-4',
        md: 'w-8 h-8',
        lg: 'w-12 h-12'
    };

    return (
        <div className="flex flex-col items-center justify-center space-y-2">
            <div
                className={`${sizeClasses[size]} ${color} animate-spin`}
                style={{
                    background: 'conic-gradient(from 0deg, currentColor, transparent 70%)',
                    borderRadius: '50%'
                }}
            >
                <div className="w-full h-full bg-white rounded-full" style={{ margin: '10%', width: '80%', height: '80%' }}></div>
            </div>
            {message && (
                <p className="text-sm text-gray-600">{message}</p>
            )}
        </div>
    );
};