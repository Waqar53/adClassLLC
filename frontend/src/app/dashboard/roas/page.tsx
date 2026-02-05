'use client';

import { useState, useEffect } from 'react';
import {
    TrendingUp,
    TrendingDown,
    DollarSign,
    Play,
    Pause,
    RefreshCw,
    Clock,
    Target,
    BarChart3,
    AlertTriangle
} from 'lucide-react';
import { roasApi, BudgetRecommendation, OptimizationResult } from '@/services/api';

interface Campaign {
    id: string;
    name: string;
    platform: string;
    currentBudget: number;
    recommendedBudget: number;
    spend: number;
    roas: number;
    predictedRoas: number;
    action: 'increase' | 'decrease' | 'maintain' | 'pause';
    confidence: number;
    reasoning: string;
}

function CampaignRow({ campaign }: { campaign: Campaign }) {
    const budgetChange = campaign.recommendedBudget - campaign.currentBudget;
    const budgetChangePercent = campaign.currentBudget > 0
        ? ((budgetChange / campaign.currentBudget) * 100).toFixed(0)
        : '0';

    const actionColors = {
        increase: 'text-success-400 bg-success-500/10',
        decrease: 'text-warning-400 bg-warning-500/10',
        maintain: 'text-neutral-400 bg-neutral-500/10',
        pause: 'text-danger-400 bg-danger-500/10'
    };

    const actionIcons = {
        increase: TrendingUp,
        decrease: TrendingDown,
        maintain: RefreshCw,
        pause: Pause
    };

    const ActionIcon = actionIcons[campaign.action];

    return (
        <div className="card p-4 hover:border-primary-500/30 transition-colors">
            <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-medium text-white truncate">{campaign.name}</h3>
                        <span className="text-xs px-2 py-0.5 rounded bg-neutral-700 text-neutral-300">
                            {campaign.platform}
                        </span>
                    </div>
                    <p className="text-sm text-neutral-500">{campaign.reasoning}</p>
                </div>

                <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-sm font-medium ${actionColors[campaign.action]}`}>
                    <ActionIcon className="w-4 h-4" />
                    <span className="capitalize">{campaign.action}</span>
                </div>
            </div>

            <div className="grid grid-cols-4 gap-4 mt-4 pt-4 border-t border-neutral-800">
                <div>
                    <p className="text-xs text-neutral-500 mb-1">Current Budget</p>
                    <p className="text-lg font-semibold text-white">${campaign.currentBudget}</p>
                </div>
                <div>
                    <p className="text-xs text-neutral-500 mb-1">Recommended</p>
                    <div className="flex items-center gap-2">
                        <p className="text-lg font-semibold text-white">${campaign.recommendedBudget}</p>
                        {budgetChange !== 0 && (
                            <span className={`text-xs ${budgetChange > 0 ? 'text-success-400' : 'text-warning-400'}`}>
                                {budgetChange > 0 ? '+' : ''}{budgetChangePercent}%
                            </span>
                        )}
                    </div>
                </div>
                <div>
                    <p className="text-xs text-neutral-500 mb-1">Current ROAS</p>
                    <p className={`text-lg font-semibold ${campaign.roas > 3 ? 'text-success-400' :
                        campaign.roas > 1 ? 'text-warning-400' : 'text-danger-400'
                        }`}>
                        {campaign.roas.toFixed(2)}x
                    </p>
                </div>
                <div>
                    <p className="text-xs text-neutral-500 mb-1">Confidence</p>
                    <p className="text-lg font-semibold text-primary-400">
                        {(campaign.confidence * 100).toFixed(0)}%
                    </p>
                </div>
            </div>

            <div className="flex gap-2 mt-4">
                <button className="btn-primary text-sm py-1.5 flex-1">
                    Apply Recommendation
                </button>
                <button className="btn-secondary text-sm py-1.5">
                    Dismiss
                </button>
            </div>
        </div>
    );
}

export default function ROASOptimizerPage() {
    const [lastOptimized, setLastOptimized] = useState('Loading...');
    const [isOptimizing, setIsOptimizing] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [campaigns, setCampaigns] = useState<Campaign[]>([]);
    const [error, setError] = useState<string | null>(null);

    // Convert API response to Campaign format
    const mapRecommendationToCampaign = (rec: BudgetRecommendation): Campaign => ({
        id: String(rec.campaign_id),
        name: rec.campaign_name,
        platform: 'Meta', // Default, could be enhanced with API
        currentBudget: rec.current_budget,
        recommendedBudget: rec.recommended_budget,
        spend: rec.current_budget * 0.7, // Estimate
        roas: rec.predicted_roas,
        predictedRoas: rec.predicted_roas,
        action: rec.action as Campaign['action'],
        confidence: rec.confidence_score,
        reasoning: rec.reasoning,
    });

    // Fetch initial data
    useEffect(() => {
        const fetchData = async () => {
            try {
                // Try to get optimization recommendations (dry run)
                const result = await roasApi.optimize({ dry_run: true });
                const mappedCampaigns = result.recommendations.map(mapRecommendationToCampaign);
                setCampaigns(mappedCampaigns);
                setLastOptimized(new Date(result.timestamp).toLocaleTimeString());
            } catch (err: any) {
                console.error('Failed to fetch ROAS data:', err);
                setError('Failed to load campaign data. Make sure the backend is running.');
            } finally {
                setIsLoading(false);
            }
        };
        fetchData();
    }, []);

    const handleOptimize = async () => {
        setIsOptimizing(true);
        setError(null);
        try {
            const result = await roasApi.optimize({ dry_run: true });
            const mappedCampaigns = result.recommendations.map(mapRecommendationToCampaign);
            setCampaigns(mappedCampaigns);
            setLastOptimized('just now');
        } catch (err: any) {
            setError(err.message || 'Optimization failed');
        } finally {
            setIsOptimizing(false);
        }
    };

    const stats = {
        totalBudget: campaigns.reduce((acc, c) => acc + c.currentBudget, 0),
        recommendedBudget: campaigns.reduce((acc, c) => acc + c.recommendedBudget, 0),
        avgRoas: campaigns.length > 0 ? campaigns.reduce((acc, c) => acc + c.roas, 0) / campaigns.length : 0,
        campaignsToAdjust: campaigns.filter(c => c.action !== 'maintain').length
    };

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-white mb-2">ROAS Optimizer</h1>
                    <p className="text-neutral-400">
                        Real-time budget optimization using Thompson Sampling & LSTM forecasting
                    </p>
                </div>

                <button
                    onClick={handleOptimize}
                    disabled={isOptimizing}
                    className="btn-primary flex items-center gap-2"
                >
                    {isOptimizing ? (
                        <>
                            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            Optimizing...
                        </>
                    ) : (
                        <>
                            <RefreshCw className="w-4 h-4" />
                            Run Optimization
                        </>
                    )}
                </button>
            </div>

            {/* Status Bar */}
            <div className="card p-4 flex items-center justify-between bg-primary-500/10 border-primary-500/30">
                <div className="flex items-center gap-3">
                    <Clock className="w-5 h-5 text-primary-400" />
                    <span className="text-sm text-primary-300">
                        Last optimization: <span className="font-medium text-white">{lastOptimized}</span>
                    </span>
                </div>
                <div className="text-sm text-primary-300">
                    Next scheduled: <span className="font-medium text-white">in 28 minutes</span>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="stat-card">
                    <div className="flex items-center justify-between">
                        <span className="metric-label">Total Daily Budget</span>
                        <DollarSign className="w-5 h-5 text-primary-400" />
                    </div>
                    <div className="metric-value text-white">${stats.totalBudget}</div>
                </div>

                <div className="stat-card">
                    <div className="flex items-center justify-between">
                        <span className="metric-label">Recommended Budget</span>
                        <Target className="w-5 h-5 text-success-400" />
                    </div>
                    <div className="metric-value text-white">${stats.recommendedBudget}</div>
                    <div className="text-sm text-success-400">
                        {stats.recommendedBudget > stats.totalBudget ? '+' : ''}
                        {((stats.recommendedBudget - stats.totalBudget) / stats.totalBudget * 100).toFixed(0)}% change
                    </div>
                </div>

                <div className="stat-card">
                    <div className="flex items-center justify-between">
                        <span className="metric-label">Average ROAS</span>
                        <BarChart3 className="w-5 h-5 text-warning-400" />
                    </div>
                    <div className="metric-value text-white">{stats.avgRoas.toFixed(2)}x</div>
                </div>

                <div className="stat-card">
                    <div className="flex items-center justify-between">
                        <span className="metric-label">Actions Pending</span>
                        <Play className="w-5 h-5 text-purple-400" />
                    </div>
                    <div className="metric-value text-white">{stats.campaignsToAdjust}</div>
                    <div className="text-sm text-neutral-500">
                        of {campaigns.length} campaigns
                    </div>
                </div>
            </div>

            {/* Campaign List */}
            <div>
                <h2 className="text-lg font-semibold text-white mb-4">Campaign Recommendations</h2>
                <div className="space-y-4">
                    {campaigns.length === 0 && !isLoading ? (
                        <div className="card p-8 text-center">
                            <p className="text-neutral-500">No campaign data available</p>
                        </div>
                    ) : campaigns.map(campaign => (
                        <CampaignRow key={campaign.id} campaign={campaign} />
                    ))}
                </div>
            </div>
        </div>
    );
}
