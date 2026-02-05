'use client';

import { useState, useEffect } from 'react';
import {
    Users,
    AlertTriangle,
    TrendingDown,
    TrendingUp,
    Activity,
    DollarSign,
    Mail,
    Phone,
    Clock,
    ChevronRight,
    Shield,
    Loader2
} from 'lucide-react';
import { churnApi, ClientHealthReport } from '@/services/api';

interface Client {
    id: string;
    name: string;
    healthScore: number;
    churnProbability: number;
    riskLevel: 'critical' | 'warning' | 'monitor' | 'healthy';
    roas: number;
    targetRoas: number;
    monthlySpend: number;
    lastContact: string;
    riskFactors: string[];
    trend: 'up' | 'down' | 'stable';
}

function HealthGauge({ score }: { score: number }) {
    const getColor = (score: number) => {
        if (score >= 80) return 'text-success-400';
        if (score >= 60) return 'text-warning-400';
        return 'text-danger-400';
    };

    return (
        <div className="relative w-24 h-24">
            <svg className="w-full h-full transform -rotate-90">
                <circle
                    cx="48"
                    cy="48"
                    r="40"
                    stroke="currentColor"
                    strokeWidth="8"
                    fill="none"
                    className="text-neutral-700"
                />
                <circle
                    cx="48"
                    cy="48"
                    r="40"
                    stroke="currentColor"
                    strokeWidth="8"
                    fill="none"
                    strokeLinecap="round"
                    strokeDasharray={`${score * 2.51} 251`}
                    className={getColor(score)}
                />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
                <span className={`text-2xl font-bold ${getColor(score)}`}>{score}</span>
            </div>
        </div>
    );
}

function ClientCard({ client }: { client: Client }) {
    const riskColors = {
        critical: 'border-l-danger-500',
        warning: 'border-l-warning-500',
        monitor: 'border-l-primary-500',
        healthy: 'border-l-success-500'
    };

    const riskBadgeColors = {
        critical: 'bg-danger-500/10 text-danger-400',
        warning: 'bg-warning-500/10 text-warning-400',
        monitor: 'bg-primary-500/10 text-primary-400',
        healthy: 'bg-success-500/10 text-success-400'
    };

    return (
        <div className={`card p-6 border-l-4 ${riskColors[client.riskLevel]} hover:bg-neutral-800/50 transition-colors cursor-pointer`}>
            <div className="flex items-start gap-6">
                <HealthGauge score={client.healthScore} />

                <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between mb-2">
                        <div>
                            <h3 className="text-lg font-semibold text-white">{client.name}</h3>
                            <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${riskBadgeColors[client.riskLevel]}`}>
                                {client.riskLevel === 'critical' && <AlertTriangle className="w-3 h-3 mr-1" />}
                                {client.riskLevel.charAt(0).toUpperCase() + client.riskLevel.slice(1)} Risk
                            </span>
                        </div>
                        <div className="text-right">
                            <p className="text-sm text-neutral-500">Churn Probability</p>
                            <p className="text-xl font-bold text-white">{(client.churnProbability * 100).toFixed(0)}%</p>
                        </div>
                    </div>

                    <div className="grid grid-cols-4 gap-4 mt-4">
                        <div>
                            <p className="text-xs text-neutral-500">ROAS</p>
                            <p className={`font-semibold ${client.roas >= client.targetRoas ? 'text-success-400' : 'text-danger-400'}`}>
                                {client.roas.toFixed(1)}x / {client.targetRoas.toFixed(1)}x
                            </p>
                        </div>
                        <div>
                            <p className="text-xs text-neutral-500">Monthly Spend</p>
                            <p className="font-semibold text-white">${client.monthlySpend.toLocaleString()}</p>
                        </div>
                        <div>
                            <p className="text-xs text-neutral-500">Last Contact</p>
                            <p className="font-semibold text-white">{client.lastContact}</p>
                        </div>
                        <div>
                            <p className="text-xs text-neutral-500">Trend</p>
                            <div className="flex items-center">
                                {client.trend === 'up' && <TrendingUp className="w-4 h-4 text-success-400" />}
                                {client.trend === 'down' && <TrendingDown className="w-4 h-4 text-danger-400" />}
                                {client.trend === 'stable' && <Activity className="w-4 h-4 text-warning-400" />}
                            </div>
                        </div>
                    </div>

                    {client.riskFactors.length > 0 && (
                        <div className="mt-4 pt-4 border-t border-neutral-800">
                            <p className="text-xs text-neutral-500 mb-2">Risk Factors:</p>
                            <div className="flex flex-wrap gap-2">
                                {client.riskFactors.map((factor, i) => (
                                    <span key={i} className="px-2 py-1 bg-neutral-800 rounded text-xs text-neutral-300">
                                        {factor}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}

                    <div className="flex gap-2 mt-4">
                        <button className="btn-primary text-sm py-1.5 flex items-center gap-1">
                            <Phone className="w-4 h-4" />
                            Schedule Call
                        </button>
                        <button className="btn-secondary text-sm py-1.5 flex items-center gap-1">
                            <Mail className="w-4 h-4" />
                            Send Email
                        </button>
                        <button className="btn-secondary text-sm py-1.5 flex items-center gap-1 ml-auto">
                            View Details
                            <ChevronRight className="w-4 h-4" />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default function ChurnPredictionPage() {
    const [filter, setFilter] = useState<string>('all');
    const [clients, setClients] = useState<Client[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Map API response to Client interface
    const mapReportToClient = (report: ClientHealthReport): Client => ({
        id: String(report.client_id),
        name: report.client_name,
        healthScore: report.health_score.overall_score,
        churnProbability: report.churn_prediction.churn_probability,
        riskLevel: report.churn_prediction.risk_level,
        roas: report.health_score.roas_score / 20, // Convert 0-100 score to 0-5 ROAS range
        targetRoas: 4.0, // Default target
        monthlySpend: 25000, // Default estimate
        lastContact: report.churn_prediction.days_to_churn ? `${report.churn_prediction.days_to_churn} days until churn` : 'Unknown',
        riskFactors: report.risk_factors.map(f => f.factor),
        trend: report.risk_factors.length > 0 ?
            (report.risk_factors[0].trend === 'declining' ? 'down' :
                report.risk_factors[0].trend === 'improving' ? 'up' : 'stable') : 'stable',
    });

    // Fetch data on mount
    useEffect(() => {
        const fetchClients = async () => {
            try {
                const data = await churnApi.getAllHealth();
                setClients(data.map(mapReportToClient));
            } catch (err: any) {
                console.error('Failed to fetch churn data:', err);
                setError('Failed to load client data. Make sure the backend is running.');
            } finally {
                setIsLoading(false);
            }
        };
        fetchClients();
    }, []);

    const stats = {
        total: clients.length,
        critical: clients.filter(c => c.riskLevel === 'critical').length,
        warning: clients.filter(c => c.riskLevel === 'warning').length,
        healthy: clients.filter(c => c.riskLevel === 'healthy').length,
        avgHealth: clients.length > 0 ? Math.round(clients.reduce((acc, c) => acc + c.healthScore, 0) / clients.length) : 0
    };

    const filteredClients = filter === 'all'
        ? clients
        : clients.filter(c => c.riskLevel === filter);

    return (
        <div className="space-y-8">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-white mb-2">Client Health & Churn Prediction</h1>
                <p className="text-neutral-400">
                    XGBoost ensemble with survival analysis for early churn detection
                </p>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
                <div className="stat-card">
                    <div className="flex items-center justify-between">
                        <span className="metric-label">Total Clients</span>
                        <Users className="w-5 h-5 text-primary-400" />
                    </div>
                    <div className="metric-value text-white">{stats.total}</div>
                </div>

                <div className="stat-card cursor-pointer hover:border-danger-500/50" onClick={() => setFilter('critical')}>
                    <div className="flex items-center justify-between">
                        <span className="metric-label">Critical</span>
                        <AlertTriangle className="w-5 h-5 text-danger-400" />
                    </div>
                    <div className="metric-value text-danger-400">{stats.critical}</div>
                </div>

                <div className="stat-card cursor-pointer hover:border-warning-500/50" onClick={() => setFilter('warning')}>
                    <div className="flex items-center justify-between">
                        <span className="metric-label">Warning</span>
                        <Clock className="w-5 h-5 text-warning-400" />
                    </div>
                    <div className="metric-value text-warning-400">{stats.warning}</div>
                </div>

                <div className="stat-card cursor-pointer hover:border-success-500/50" onClick={() => setFilter('healthy')}>
                    <div className="flex items-center justify-between">
                        <span className="metric-label">Healthy</span>
                        <Shield className="w-5 h-5 text-success-400" />
                    </div>
                    <div className="metric-value text-success-400">{stats.healthy}</div>
                </div>

                <div className="stat-card">
                    <div className="flex items-center justify-between">
                        <span className="metric-label">Avg Health Score</span>
                        <Activity className="w-5 h-5 text-primary-400" />
                    </div>
                    <div className="metric-value text-white">{stats.avgHealth}</div>
                </div>
            </div>

            {/* Filter Tabs */}
            <div className="flex gap-2">
                {['all', 'critical', 'warning', 'monitor', 'healthy'].map(level => (
                    <button
                        key={level}
                        onClick={() => setFilter(level)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${filter === level
                            ? 'bg-primary-500/20 text-primary-400'
                            : 'bg-neutral-800 text-neutral-400 hover:text-white'
                            }`}
                    >
                        {level.charAt(0).toUpperCase() + level.slice(1)}
                    </button>
                ))}
            </div>

            {/* Client List */}
            <div className="space-y-4">
                {filteredClients.map(client => (
                    <ClientCard key={client.id} client={client} />
                ))}
            </div>
        </div>
    );
}
