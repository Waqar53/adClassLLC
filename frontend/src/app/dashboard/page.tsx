'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import {
    Sparkles,
    TrendingUp,
    Users,
    AlertTriangle,
    ArrowUpRight,
    Brain,
    DollarSign,
    Target,
    GitBranch,
    Layers,
    ChevronRight,
    RefreshCw,
} from 'lucide-react';
import { dashboardApi, DashboardStat, TopCampaign, DashboardAlert, DashboardSummary, ModelMetrics } from '@/services/api';

// Icon mapping for stats
const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
    'Total ROAS': DollarSign,
    'Active Campaigns': Target,
    'Clients': Users,
    'At-Risk Clients': AlertTriangle,
};

// Module cards with icons
const modules = [
    {
        title: 'Creative Performance',
        description: 'Predict ad creative performance before spending (85% accuracy)',
        accuracy: 0.85,
        icon: Sparkles,
        href: '/dashboard/creative',
        gradient: 'from-purple-500 to-pink-500',
    },
    {
        title: 'ROAS Optimizer',
        description: 'Real-time budget optimization using reinforcement learning',
        accuracy: 0.89,
        icon: TrendingUp,
        href: '/dashboard/roas',
        gradient: 'from-green-500 to-emerald-500',
    },
    {
        title: 'Churn Prediction',
        description: 'Predict client churn 60 days early for early intervention',
        accuracy: 0.82,
        icon: Users,
        href: '/dashboard/churn',
        gradient: 'from-orange-500 to-amber-500',
    },
    {
        title: 'Attribution Engine',
        description: 'Shapley value-based multi-touch attribution',
        accuracy: 0.91,
        icon: GitBranch,
        href: '/dashboard/attribution',
        gradient: 'from-blue-500 to-cyan-500',
    },
    {
        title: 'Audience Intelligence',
        description: 'AI-powered high-converting audience segments',
        accuracy: 0.87,
        icon: Layers,
        href: '/dashboard/audience',
        gradient: 'from-violet-500 to-purple-500',
    },
];

export default function DashboardPage() {
    const [summary, setSummary] = useState<DashboardSummary | null>(null);
    const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchDashboardData = async () => {
        setLoading(true);
        setError(null);
        try {
            const [summaryData, metricsData] = await Promise.all([
                dashboardApi.getSummary(),
                dashboardApi.getModelMetrics(),
            ]);
            setSummary(summaryData);
            setModelMetrics(metricsData);
        } catch (err: any) {
            console.error('Failed to fetch dashboard data:', err);
            setError(err.message || 'Failed to load dashboard data');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchDashboardData();
    }, []);

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-900 flex items-center justify-center">
                <div className="text-center">
                    <RefreshCw className="w-12 h-12 text-purple-500 animate-spin mx-auto mb-4" />
                    <p className="text-gray-400">Loading real-time data from ML models...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-screen bg-gray-900 flex items-center justify-center">
                <div className="text-center max-w-md">
                    <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                    <p className="text-red-400 mb-4">{error}</p>
                    <button
                        onClick={fetchDashboardData}
                        className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                        <Brain className="w-8 h-8 text-purple-500" />
                        AI Dashboard
                    </h1>
                    <p className="text-gray-400 mt-1">
                        Real-time insights powered by 5 ML models • Live data from Google Ads, Meta & TikTok
                    </p>
                </div>
                <button
                    onClick={fetchDashboardData}
                    className="flex items-center gap-2 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-300 hover:bg-gray-700 transition"
                >
                    <RefreshCw className="w-4 h-4" />
                    Refresh
                </button>
            </div>

            {/* Stats Grid - Real Data */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {summary?.stats.map((stat, index) => {
                    const Icon = iconMap[stat.label] || TrendingUp;
                    return (
                        <div
                            key={index}
                            className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6 hover:border-purple-500/50 transition"
                        >
                            <div className="flex items-center justify-between mb-4">
                                <div className="p-3 bg-purple-500/10 rounded-lg">
                                    <Icon className="w-6 h-6 text-purple-400" />
                                </div>
                                <span
                                    className={`text-sm font-medium flex items-center gap-1 ${stat.change_type === 'positive'
                                            ? 'text-green-400'
                                            : stat.change_type === 'negative'
                                                ? 'text-red-400'
                                                : 'text-gray-400'
                                        }`}
                                >
                                    {stat.change}
                                    {stat.change_type !== 'neutral' && <ArrowUpRight className="w-4 h-4" />}
                                </span>
                            </div>
                            <p className="text-3xl font-bold text-white mb-1">{stat.value}</p>
                            <p className="text-gray-400 text-sm">{stat.label}</p>
                        </div>
                    );
                })}
            </div>

            {/* Module Cards */}
            <div>
                <h2 className="text-xl font-semibold text-white mb-4">AI Modules</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {modules.map((module, index) => {
                        const Icon = module.icon;
                        // Get real accuracy from model metrics if available
                        const realAccuracy = modelMetrics
                            ? module.title === 'Creative Performance'
                                ? modelMetrics.creative_predictor.accuracy
                                : module.title === 'ROAS Optimizer'
                                    ? modelMetrics.roas_optimizer.accuracy
                                    : module.title === 'Churn Prediction'
                                        ? modelMetrics.churn_predictor.accuracy
                                        : module.title === 'Attribution Engine'
                                            ? modelMetrics.attribution_engine.shapley_confidence
                                            : modelMetrics.audience_segmenter.silhouette_score
                            : module.accuracy;

                        return (
                            <Link
                                key={index}
                                href={module.href}
                                className="group bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6 hover:border-purple-500/50 transition-all hover:-translate-y-1"
                            >
                                <div className="flex items-start justify-between mb-4">
                                    <div className={`p-3 rounded-lg bg-gradient-to-r ${module.gradient}`}>
                                        <Icon className="w-6 h-6 text-white" />
                                    </div>
                                    <span className="text-sm font-medium text-green-400 bg-green-400/10 px-2 py-1 rounded">
                                        {(realAccuracy * 100).toFixed(0)}% accuracy
                                    </span>
                                </div>
                                <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-purple-400 transition">
                                    {module.title}
                                </h3>
                                <p className="text-gray-400 text-sm mb-4">{module.description}</p>
                                <div className="flex items-center text-purple-400 text-sm">
                                    Open Module <ChevronRight className="w-4 h-4 ml-1" />
                                </div>
                            </Link>
                        );
                    })}
                </div>
            </div>

            {/* Two Column Layout: Top Campaigns & Recent Alerts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Top Campaigns - Real Data */}
                <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-xl font-semibold text-white">Top Campaigns</h2>
                        <Link
                            href="/dashboard/roas"
                            className="text-purple-400 text-sm hover:text-purple-300 flex items-center gap-1"
                        >
                            View All <ChevronRight className="w-4 h-4" />
                        </Link>
                    </div>
                    <div className="space-y-4">
                        {summary?.top_campaigns.map((campaign, index) => (
                            <div
                                key={index}
                                className="flex items-center justify-between p-4 bg-gray-900/50 rounded-lg border border-gray-700"
                            >
                                <div className="flex items-center gap-3">
                                    <div className="w-2 h-2 rounded-full bg-green-400"></div>
                                    <div>
                                        <p className="text-white font-medium">{campaign.name}</p>
                                        <p className="text-gray-400 text-sm">
                                            Spend: ${campaign.spend.toLocaleString()}
                                        </p>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className="text-white font-semibold text-lg">{campaign.roas.toFixed(2)}x</p>
                                    <p className="text-green-400 text-sm">ROAS</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Recent Alerts - Real Data */}
                <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-xl font-semibold text-white">AI Alerts</h2>
                        <Link
                            href="/dashboard/churn"
                            className="text-purple-400 text-sm hover:text-purple-300 flex items-center gap-1"
                        >
                            View All <ChevronRight className="w-4 h-4" />
                        </Link>
                    </div>
                    <div className="space-y-4">
                        {summary?.recent_alerts.map((alert, index) => (
                            <div
                                key={index}
                                className={`p-4 rounded-lg border ${alert.type === 'warning'
                                        ? 'bg-amber-500/10 border-amber-500/20'
                                        : alert.type === 'success'
                                            ? 'bg-green-500/10 border-green-500/20'
                                            : 'bg-blue-500/10 border-blue-500/20'
                                    }`}
                            >
                                <div className="flex items-start gap-3">
                                    <AlertTriangle
                                        className={`w-5 h-5 ${alert.type === 'warning'
                                                ? 'text-amber-400'
                                                : alert.type === 'success'
                                                    ? 'text-green-400'
                                                    : 'text-blue-400'
                                            }`}
                                    />
                                    <div className="flex-1">
                                        <p className="text-white text-sm">{alert.message}</p>
                                        <p className="text-gray-400 text-xs mt-1">{alert.time}</p>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Model Performance Footer */}
            <div className="bg-gradient-to-r from-purple-900/50 to-pink-900/50 border border-purple-500/20 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                    <Brain className="w-6 h-6 text-purple-400" />
                    <h2 className="text-lg font-semibold text-white">ML Model Performance</h2>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    {modelMetrics && (
                        <>
                            <div className="text-center">
                                <p className="text-2xl font-bold text-white">
                                    {(modelMetrics.creative_predictor.accuracy * 100).toFixed(0)}%
                                </p>
                                <p className="text-gray-400 text-sm">Creative Predictor</p>
                                <p className="text-gray-500 text-xs">
                                    {modelMetrics.creative_predictor.predictions_made.toLocaleString()} predictions
                                </p>
                            </div>
                            <div className="text-center">
                                <p className="text-2xl font-bold text-white">
                                    {(modelMetrics.roas_optimizer.accuracy * 100).toFixed(0)}%
                                </p>
                                <p className="text-gray-400 text-sm">ROAS Optimizer</p>
                                <p className="text-gray-500 text-xs">
                                    +{(modelMetrics.roas_optimizer.avg_roas_improvement * 100).toFixed(0)}% avg improvement
                                </p>
                            </div>
                            <div className="text-center">
                                <p className="text-2xl font-bold text-white">
                                    {modelMetrics.churn_predictor.early_detection_days}d
                                </p>
                                <p className="text-gray-400 text-sm">Early Detection</p>
                                <p className="text-gray-500 text-xs">
                                    {modelMetrics.churn_predictor.clients_monitored} clients monitored
                                </p>
                            </div>
                            <div className="text-center">
                                <p className="text-2xl font-bold text-white">
                                    {(modelMetrics.attribution_engine.shapley_confidence * 100).toFixed(0)}%
                                </p>
                                <p className="text-gray-400 text-sm">Attribution Confidence</p>
                                <p className="text-gray-500 text-xs">
                                    {modelMetrics.attribution_engine.journeys_analyzed.toLocaleString()} journeys
                                </p>
                            </div>
                            <div className="text-center">
                                <p className="text-2xl font-bold text-white">
                                    {modelMetrics.audience_segmenter.segments_created}
                                </p>
                                <p className="text-gray-400 text-sm">Segments Created</p>
                                <p className="text-gray-500 text-xs">
                                    {(modelMetrics.audience_segmenter.users_segmented / 1000).toFixed(0)}k users
                                </p>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
