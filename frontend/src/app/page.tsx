'use client';

import Link from 'next/link';
import {
    ArrowRight,
    BarChart3,
    Brain,
    Target,
    Users,
    TrendingUp,
    Zap,
    Shield
} from 'lucide-react';

const features = [
    {
        icon: Brain,
        title: 'Creative Predictor',
        description: 'Predict ad performance before launch with 95% accuracy using multi-modal AI',
        href: '/dashboard/creative',
        color: 'text-purple-500',
    },
    {
        icon: TrendingUp,
        title: 'ROAS Optimizer',
        description: 'Real-time budget optimization using Thompson Sampling & LSTM forecasting',
        href: '/dashboard/roas',
        color: 'text-green-500',
    },
    {
        icon: Users,
        title: 'Churn Prediction',
        description: 'Identify at-risk clients 45 days early with XGBoost survival analysis',
        href: '/dashboard/churn',
        color: 'text-orange-500',
    },
    {
        icon: BarChart3,
        title: 'Attribution Engine',
        description: 'Shapley value-based multi-touch attribution across all channels',
        href: '/dashboard/attribution',
        color: 'text-blue-500',
    },
    {
        icon: Target,
        title: 'Audience Intelligence',
        description: 'AI-powered segmentation and lookalike audience generation',
        href: '/dashboard/audience',
        color: 'text-pink-500',
    },
];

const stats = [
    { value: '+47%', label: 'ROAS Improvement' },
    { value: '-32%', label: 'CPA Reduction' },
    { value: '45 days', label: 'Early Churn Detection' },
    { value: '3.5x', label: 'Account Manager Capacity' },
];

export default function Home() {
    return (
        <main className="min-h-screen">
            {/* Hero Section */}
            <section className="relative overflow-hidden">
                {/* Background gradient */}
                <div className="absolute inset-0 bg-gradient-to-br from-primary-900/20 via-neutral-950 to-neutral-950" />
                <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary-500/10 via-transparent to-transparent" />

                {/* Content */}
                <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
                    <div className="text-center max-w-4xl mx-auto">
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary-500/10 border border-primary-500/20 text-primary-400 text-sm mb-8">
                            <Zap className="w-4 h-4" />
                            AI-Powered Ad Intelligence
                        </div>

                        <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tight mb-6">
                            <span className="text-white">Turn Ad Data Into</span>
                            <br />
                            <span className="text-gradient">Predictable Growth</span>
                        </h1>

                        <p className="text-xl text-neutral-400 mb-10 max-w-2xl mx-auto">
                            The AI system that predicts creative performance, optimizes budgets in real-time,
                            and prevents client churn before it happens.
                        </p>

                        <div className="flex flex-col sm:flex-row gap-4 justify-center">
                            <Link
                                href="/dashboard"
                                className="btn-primary inline-flex items-center justify-center gap-2 text-lg px-8 py-3"
                            >
                                Open Dashboard
                                <ArrowRight className="w-5 h-5" />
                            </Link>
                            <Link
                                href="#features"
                                className="btn-secondary inline-flex items-center justify-center gap-2 text-lg px-8 py-3"
                            >
                                Explore Features
                            </Link>
                        </div>
                    </div>
                </div>
            </section>

            {/* Stats Section */}
            <section className="py-16 border-y border-neutral-800">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                        {stats.map((stat, i) => (
                            <div key={i} className="text-center">
                                <div className="text-4xl font-bold text-white mb-2">{stat.value}</div>
                                <div className="text-sm text-neutral-500">{stat.label}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section id="features" className="py-24">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
                            5 AI Modules. Infinite Possibilities.
                        </h2>
                        <p className="text-lg text-neutral-400 max-w-2xl mx-auto">
                            Each module is designed to solve a specific challenge in ad management,
                            powered by state-of-the-art machine learning.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {features.map((feature, i) => (
                            <Link
                                key={i}
                                href={feature.href}
                                className="card p-6 hover:border-primary-500/50 transition-all duration-300 group"
                            >
                                <feature.icon className={`w-10 h-10 ${feature.color} mb-4`} />
                                <h3 className="text-xl font-semibold text-white mb-2 group-hover:text-primary-400 transition-colors">
                                    {feature.title}
                                </h3>
                                <p className="text-neutral-400 text-sm">
                                    {feature.description}
                                </p>
                                <div className="mt-4 text-primary-400 text-sm font-medium flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                    Explore <ArrowRight className="w-4 h-4" />
                                </div>
                            </Link>
                        ))}

                        {/* More coming */}
                        <div className="card p-6 border-dashed flex flex-col items-center justify-center text-center">
                            <Shield className="w-10 h-10 text-neutral-600 mb-4" />
                            <h3 className="text-xl font-semibold text-neutral-500 mb-2">
                                More Coming Soon
                            </h3>
                            <p className="text-neutral-600 text-sm">
                                Additional AI modules in development
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="py-24 border-t border-neutral-800">
                <div className="max-w-4xl mx-auto px-4 text-center">
                    <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
                        Ready to Transform Your Ad Operations?
                    </h2>
                    <p className="text-lg text-neutral-400 mb-8">
                        Start predicting, optimizing, and growing today.
                    </p>
                    <Link
                        href="/dashboard"
                        className="btn-primary inline-flex items-center justify-center gap-2 text-lg px-8 py-3"
                    >
                        Get Started
                        <ArrowRight className="w-5 h-5" />
                    </Link>
                </div>
            </section>
        </main>
    );
}
