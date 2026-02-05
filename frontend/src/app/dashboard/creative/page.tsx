'use client';

import { useState } from 'react';
import {
    Upload,
    Image as ImageIcon,
    FileText,
    Sparkles,
    AlertCircle,
    CheckCircle,
    TrendingUp,
    TrendingDown,
    Info,
    AlertTriangle
} from 'lucide-react';
import { creativeApi, CreativePrediction } from '@/services/api';

// Prediction result type from API
interface PredictionResult extends CreativePrediction { }

function ScoreCard({
    label,
    score,
    maxScore = 100,
    description
}: {
    label: string;
    score: number;
    maxScore?: number;
    description?: string;
}) {
    const percentage = (score / maxScore) * 100;
    const color = percentage >= 70 ? 'bg-success-500' : percentage >= 40 ? 'bg-warning-500' : 'bg-danger-500';

    return (
        <div className="p-4 bg-neutral-800/50 rounded-lg">
            <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-neutral-400">{label}</span>
                <span className="text-lg font-semibold text-white">{score.toFixed(0)}</span>
            </div>
            <div className="w-full h-2 bg-neutral-700 rounded-full overflow-hidden">
                <div
                    className={`h-full ${color} transition-all duration-500`}
                    style={{ width: `${percentage}%` }}
                />
            </div>
            {description && (
                <p className="text-xs text-neutral-500 mt-2">{description}</p>
            )}
        </div>
    );
}

export default function CreativePredictorPage() {
    const [headline, setHeadline] = useState('');
    const [bodyText, setBodyText] = useState('');
    const [ctaType, setCtaType] = useState('SHOP_NOW');
    const [platform, setPlatform] = useState('meta');
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handlePredict = async () => {
        if (!headline) return;

        setIsLoading(true);
        setError(null);

        try {
            const prediction = await creativeApi.predict({
                headline,
                body_text: bodyText || undefined,
                cta_type: ctaType,
                platform,
            });
            setResult(prediction);
        } catch (err: any) {
            setError(err.message || 'Failed to predict creative performance');
            console.error('Prediction error:', err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="space-y-8">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-white mb-2">Creative Performance Predictor</h1>
                <p className="text-neutral-400">
                    Predict ad performance before launch using multi-modal AI
                </p>
            </div>

            <div className="grid lg:grid-cols-2 gap-8">
                {/* Input Section */}
                <div className="space-y-6">
                    <div className="card p-6">
                        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <FileText className="w-5 h-5 text-purple-400" />
                            Ad Content
                        </h2>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-neutral-300 mb-2">
                                    Headline *
                                </label>
                                <input
                                    type="text"
                                    value={headline}
                                    onChange={(e) => setHeadline(e.target.value)}
                                    placeholder="Enter your ad headline..."
                                    className="input"
                                    maxLength={125}
                                />
                                <p className="text-xs text-neutral-500 mt-1">{headline.length}/125 characters</p>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-neutral-300 mb-2">
                                    Body Text
                                </label>
                                <textarea
                                    value={bodyText}
                                    onChange={(e) => setBodyText(e.target.value)}
                                    placeholder="Enter your ad body text..."
                                    className="input min-h-[100px] resize-none"
                                    maxLength={500}
                                />
                                <p className="text-xs text-neutral-500 mt-1">{bodyText.length}/500 characters</p>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium text-neutral-300 mb-2">
                                        CTA Type
                                    </label>
                                    <select
                                        value={ctaType}
                                        onChange={(e) => setCtaType(e.target.value)}
                                        className="input"
                                    >
                                        <option value="SHOP_NOW">Shop Now</option>
                                        <option value="LEARN_MORE">Learn More</option>
                                        <option value="SIGN_UP">Sign Up</option>
                                        <option value="BUY_NOW">Buy Now</option>
                                        <option value="GET_OFFER">Get Offer</option>
                                        <option value="BOOK_NOW">Book Now</option>
                                    </select>
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-neutral-300 mb-2">
                                        Platform
                                    </label>
                                    <select
                                        value={platform}
                                        onChange={(e) => setPlatform(e.target.value)}
                                        className="input"
                                    >
                                        <option value="meta">Meta (Facebook/Instagram)</option>
                                        <option value="google">Google Ads</option>
                                        <option value="tiktok">TikTok</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Image Upload */}
                    <div className="card p-6">
                        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <ImageIcon className="w-5 h-5 text-purple-400" />
                            Creative Assets (Optional)
                        </h2>

                        <div className="border-2 border-dashed border-neutral-700 rounded-lg p-8 text-center hover:border-primary-500/50 transition-colors cursor-pointer">
                            <Upload className="w-10 h-10 text-neutral-500 mx-auto mb-3" />
                            <p className="text-neutral-400 text-sm mb-1">
                                Drag & drop an image or video
                            </p>
                            <p className="text-neutral-500 text-xs">
                                PNG, JPG, MP4 up to 50MB
                            </p>
                        </div>
                    </div>

                    <button
                        onClick={handlePredict}
                        disabled={!headline || isLoading}
                        className="btn-primary w-full flex items-center justify-center gap-2 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isLoading ? (
                            <>
                                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                Analyzing...
                            </>
                        ) : (
                            <>
                                <Sparkles className="w-5 h-5" />
                                Predict Performance
                            </>
                        )}
                    </button>
                </div>

                {/* Results Section */}
                <div className="space-y-6">
                    {/* Error Display */}
                    {error && (
                        <div className="card p-4 bg-danger-500/10 border-danger-500/30">
                            <div className="flex items-center gap-3">
                                <AlertTriangle className="w-5 h-5 text-danger-400" />
                                <div>
                                    <p className="text-danger-400 font-medium">Error</p>
                                    <p className="text-sm text-neutral-400">{error}</p>
                                </div>
                            </div>
                        </div>
                    )}

                    {!result && !isLoading && !error && (
                        <div className="card p-12 text-center">
                            <Sparkles className="w-12 h-12 text-neutral-600 mx-auto mb-4" />
                            <h3 className="text-lg font-medium text-neutral-400 mb-2">
                                No Prediction Yet
                            </h3>
                            <p className="text-neutral-500 text-sm">
                                Enter your ad content and click "Predict Performance" to get AI insights
                            </p>
                        </div>
                    )}

                    {result && (
                        <>
                            {/* Main Metrics */}
                            <div className="grid grid-cols-2 gap-4">
                                <div className="card p-6 text-center border-l-4 border-l-success-500">
                                    <p className="text-sm text-neutral-400 mb-1">Predicted CTR</p>
                                    <p className="text-3xl font-bold text-white">{result.predicted_ctr.toFixed(2)}%</p>
                                    <div className="flex items-center justify-center gap-1 text-success-400 text-sm mt-2">
                                        <TrendingUp className="w-4 h-4" />
                                        <span>+12% vs avg</span>
                                    </div>
                                </div>

                                <div className="card p-6 text-center border-l-4 border-l-primary-500">
                                    <p className="text-sm text-neutral-400 mb-1">Predicted CVR</p>
                                    <p className="text-3xl font-bold text-white">{result.predicted_cvr.toFixed(2)}%</p>
                                    <div className="flex items-center justify-center gap-1 text-primary-400 text-sm mt-2">
                                        <Info className="w-4 h-4" />
                                        <span>Industry avg: 4.2%</span>
                                    </div>
                                </div>
                            </div>

                            {/* Quality Scores */}
                            <div className="card p-6">
                                <h3 className="text-lg font-semibold text-white mb-4">Quality Scores</h3>
                                <div className="grid grid-cols-2 gap-4">
                                    <ScoreCard
                                        label="Overall Quality"
                                        score={result.overall_quality_score}
                                        description="Combined score from all factors"
                                    />
                                    <ScoreCard
                                        label="Hook Strength"
                                        score={result.hook_strength_score}
                                        description="Attention-grabbing potential"
                                    />
                                    <ScoreCard
                                        label="Readability"
                                        score={result.readability_score}
                                        description="How easy to read and understand"
                                    />
                                    <ScoreCard
                                        label="Brand Consistency"
                                        score={result.brand_consistency_score}
                                        description="Alignment with brand guidelines"
                                    />
                                </div>
                            </div>

                            {/* Recommendations */}
                            <div className="card p-6">
                                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                    <AlertCircle className="w-5 h-5 text-warning-400" />
                                    Recommendations
                                </h3>
                                <ul className="space-y-3">
                                    {result.recommendations.map((rec, i) => (
                                        <li key={i} className="flex items-start gap-3 text-sm">
                                            <CheckCircle className="w-5 h-5 text-success-400 flex-shrink-0 mt-0.5" />
                                            <span className="text-neutral-300">{rec}</span>
                                        </li>
                                    ))}
                                </ul>
                            </div>

                            {/* Confidence */}
                            <div className="card p-4 bg-primary-500/10 border-primary-500/30">
                                <div className="flex items-center justify-between">
                                    <span className="text-sm text-primary-300">Prediction Confidence</span>
                                    <span className="text-lg font-semibold text-white">
                                        {(result.confidence_score * 100).toFixed(0)}%
                                    </span>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
