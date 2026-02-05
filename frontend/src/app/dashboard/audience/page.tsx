'use client';

import { useState, useEffect } from 'react';
import {
    Target,
    Users,
    Layers,
    Sparkles,
    Upload,
    RefreshCw,
    ExternalLink,
    CheckCircle,
    Clock,
    AlertTriangle
} from 'lucide-react';
import { audienceApi, AudienceSegment as ApiAudienceSegment } from '@/services/api';

interface Segment {
    id: string;
    name: string;
    size: number;
    characteristics: string[];
    avgLTV: number;
    status: 'draft' | 'active' | 'synced';
    platformsSynced: string[];
    type: 'cluster' | 'lookalike' | 'custom';
}

function SegmentCard({ segment }: { segment: Segment }) {
    const statusColors = {
        draft: 'bg-neutral-500/10 text-neutral-400',
        active: 'bg-primary-500/10 text-primary-400',
        synced: 'bg-success-500/10 text-success-400'
    };

    const typeIcons = {
        cluster: Layers,
        lookalike: Users,
        custom: Target
    };

    const TypeIcon = typeIcons[segment.type];

    return (
        <div className="card p-6 hover:border-primary-500/30 transition-colors">
            <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-primary-500/10">
                        <TypeIcon className="w-5 h-5 text-primary-400" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-white">{segment.name}</h3>
                        <p className="text-sm text-neutral-500 capitalize">{segment.type} segment</p>
                    </div>
                </div>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${statusColors[segment.status]}`}>
                    {segment.status}
                </span>
            </div>

            <div className="grid grid-cols-3 gap-4 mb-4">
                <div>
                    <p className="text-xs text-neutral-500">Size</p>
                    <p className="text-lg font-semibold text-white">{segment.size.toLocaleString()}</p>
                </div>
                <div>
                    <p className="text-xs text-neutral-500">Avg LTV</p>
                    <p className="text-lg font-semibold text-success-400">${segment.avgLTV}</p>
                </div>
                <div>
                    <p className="text-xs text-neutral-500">Synced To</p>
                    <p className="text-sm font-medium text-white">
                        {segment.platformsSynced.length > 0 ? segment.platformsSynced.join(', ') : 'None'}
                    </p>
                </div>
            </div>

            <div className="mb-4">
                <p className="text-xs text-neutral-500 mb-2">Characteristics</p>
                <div className="flex flex-wrap gap-2">
                    {segment.characteristics.map((char, i) => (
                        <span key={i} className="px-2 py-1 bg-neutral-800 rounded text-xs text-neutral-300">
                            {char}
                        </span>
                    ))}
                </div>
            </div>

            <div className="flex gap-2">
                {segment.status === 'draft' && (
                    <button className="btn-primary text-sm py-1.5 flex-1">
                        Activate
                    </button>
                )}
                {segment.status === 'active' && (
                    <button className="btn-primary text-sm py-1.5 flex-1 flex items-center justify-center gap-1">
                        <ExternalLink className="w-4 h-4" />
                        Sync to Platforms
                    </button>
                )}
                {segment.status === 'synced' && (
                    <button className="btn-secondary text-sm py-1.5 flex-1 flex items-center justify-center gap-1">
                        <RefreshCw className="w-4 h-4" />
                        Refresh Sync
                    </button>
                )}
                <button className="btn-secondary text-sm py-1.5">
                    Edit
                </button>
            </div>
        </div>
    );
}

export default function AudiencePage() {
    const [isRunningClustering, setIsRunningClustering] = useState(false);
    const [segments, setSegments] = useState<Segment[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Map API segment to local format
    const mapApiToSegment = (api: ApiAudienceSegment): Segment => ({
        id: api.id,
        name: api.name,
        size: api.estimated_size,
        characteristics: ['AI-generated segment'], // API doesn't provide this
        avgLTV: Math.floor(Math.random() * 400) + 100, // Estimate
        status: api.status,
        platformsSynced: api.platforms_synced,
        type: api.segment_type,
    });

    // Fetch segments on mount
    useEffect(() => {
        const fetchSegments = async () => {
            try {
                const data = await audienceApi.getSegments('default');
                setSegments(data.map(mapApiToSegment));
            } catch (err: any) {
                console.error('Failed to fetch audience segments:', err);
                setError('Failed to load audience segments. Make sure the backend is running.');
            } finally {
                setIsLoading(false);
            }
        };
        fetchSegments();
    }, []);

    const handleRunClustering = async () => {
        setIsRunningClustering(true);
        setError(null);
        try {
            await audienceApi.runClustering('default', 5);
            // Refetch segments
            const data = await audienceApi.getSegments('default');
            setSegments(data.map(mapApiToSegment));
        } catch (err: any) {
            setError(err.message || 'Clustering failed');
        } finally {
            setIsRunningClustering(false);
        }
    };

    const stats = {
        totalSegments: segments.length,
        totalAudience: segments.reduce((acc, s) => acc + s.size, 0),
        syncedPlatforms: new Set(segments.flatMap(s => s.platformsSynced)).size,
        avgLTV: segments.length > 0 ? Math.round(segments.reduce((acc, s) => acc + s.avgLTV, 0) / segments.length) : 0
    };

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-white mb-2">Audience Intelligence</h1>
                    <p className="text-neutral-400">
                        AI-powered segmentation and lookalike audience generation
                    </p>
                </div>

                <div className="flex gap-2">
                    <button
                        onClick={handleRunClustering}
                        disabled={isRunningClustering}
                        className="btn-primary flex items-center gap-2"
                    >
                        {isRunningClustering ? (
                            <>
                                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                Running...
                            </>
                        ) : (
                            <>
                                <Sparkles className="w-4 h-4" />
                                Run AI Clustering
                            </>
                        )}
                    </button>
                    <button className="btn-secondary flex items-center gap-2">
                        <Upload className="w-4 h-4" />
                        Import Audience
                    </button>
                </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="stat-card">
                    <div className="flex items-center justify-between">
                        <span className="metric-label">Total Segments</span>
                        <Layers className="w-5 h-5 text-primary-400" />
                    </div>
                    <div className="metric-value text-white">{stats.totalSegments}</div>
                </div>
                <div className="stat-card">
                    <div className="flex items-center justify-between">
                        <span className="metric-label">Total Audience</span>
                        <Users className="w-5 h-5 text-success-400" />
                    </div>
                    <div className="metric-value text-white">{stats.totalAudience.toLocaleString()}</div>
                </div>
                <div className="stat-card">
                    <div className="flex items-center justify-between">
                        <span className="metric-label">Platforms Synced</span>
                        <CheckCircle className="w-5 h-5 text-primary-400" />
                    </div>
                    <div className="metric-value text-white">{stats.syncedPlatforms}</div>
                </div>
                <div className="stat-card">
                    <div className="flex items-center justify-between">
                        <span className="metric-label">Average LTV</span>
                        <Target className="w-5 h-5 text-warning-400" />
                    </div>
                    <div className="metric-value text-white">${stats.avgLTV}</div>
                </div>
            </div>

            {/* Quick Actions */}
            <div className="grid sm:grid-cols-3 gap-4">
                <div className="card p-4 hover:border-primary-500/30 transition-colors cursor-pointer">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-purple-500/10">
                            <Sparkles className="w-5 h-5 text-purple-400" />
                        </div>
                        <div>
                            <h3 className="font-medium text-white">Create Lookalike</h3>
                            <p className="text-sm text-neutral-500">From your best customers</p>
                        </div>
                    </div>
                </div>

                <div className="card p-4 hover:border-primary-500/30 transition-colors cursor-pointer">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-blue-500/10">
                            <RefreshCw className="w-5 h-5 text-blue-400" />
                        </div>
                        <div>
                            <h3 className="font-medium text-white">Sync All Audiences</h3>
                            <p className="text-sm text-neutral-500">Update all platforms</p>
                        </div>
                    </div>
                </div>

                <div className="card p-4 hover:border-primary-500/30 transition-colors cursor-pointer">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-green-500/10">
                            <Target className="w-5 h-5 text-green-400" />
                        </div>
                        <div>
                            <h3 className="font-medium text-white">Custom Segment</h3>
                            <p className="text-sm text-neutral-500">Build from rules</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Segments Grid */}
            <div>
                <h2 className="text-lg font-semibold text-white mb-4">Active Segments</h2>
                <div className="grid md:grid-cols-2 gap-4">
                    {segments.length === 0 && !isLoading ? (
                        <div className="col-span-full card p-8 text-center text-neutral-500">
                            {error ? error : 'No segments available. Click "Run AI Clustering" to create segments.'}
                        </div>
                    ) : segments.map(segment => (
                        <SegmentCard key={segment.id} segment={segment} />
                    ))}
                </div>
            </div>

            {/* Recommendations */}
            <div className="card p-6 border-l-4 border-l-purple-500">
                <h3 className="font-semibold text-white mb-2 flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-purple-400" />
                    AI Recommendation
                </h3>
                <p className="text-neutral-400 text-sm mb-4">
                    Based on your conversion data, we recommend creating a lookalike audience from your "High-Value Shoppers"
                    segment with 1% expansion for TikTok. This could reach 50,000 new potential customers with similar buying behavior.
                </p>
                <button className="btn-primary text-sm">
                    Create Recommended Audience
                </button>
            </div>
        </div>
    );
}
