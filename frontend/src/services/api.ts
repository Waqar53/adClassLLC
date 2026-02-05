/**
 * API Service Layer
 * 
 * Centralized API client for communicating with the backend.
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

// Types
export interface ApiError {
    status: number;
    message: string;
    detail?: string;
}

export interface PaginatedResponse<T> {
    items: T[];
    total: number;
    page: number;
    page_size: number;
}

// API Client
class ApiClient {
    private client: AxiosInstance;

    constructor() {
        this.client = axios.create({
            baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1',
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
            },
        });

        // Request interceptor for auth
        this.client.interceptors.request.use(
            (config) => {
                const token = typeof window !== 'undefined'
                    ? localStorage.getItem('auth_token')
                    : null;
                if (token) {
                    config.headers.Authorization = `Bearer ${token}`;
                }
                return config;
            },
            (error) => Promise.reject(error)
        );

        // Response interceptor for error handling
        this.client.interceptors.response.use(
            (response) => response,
            (error: AxiosError) => {
                const apiError: ApiError = {
                    status: error.response?.status || 500,
                    message: error.message,
                    detail: (error.response?.data as any)?.detail,
                };
                return Promise.reject(apiError);
            }
        );
    }

    // Generic request methods
    async get<T>(url: string, params?: object): Promise<T> {
        const response = await this.client.get<T>(url, { params });
        return response.data;
    }

    async post<T>(url: string, data?: object): Promise<T> {
        const response = await this.client.post<T>(url, data);
        return response.data;
    }

    async put<T>(url: string, data?: object): Promise<T> {
        const response = await this.client.put<T>(url, data);
        return response.data;
    }

    async delete<T>(url: string): Promise<T> {
        const response = await this.client.delete<T>(url);
        return response.data;
    }

    async upload<T>(url: string, formData: FormData): Promise<T> {
        const response = await this.client.post<T>(url, formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return response.data;
    }
}

const api = new ApiClient();

// ===========================================
// CREATIVE PREDICTOR SERVICE
// ===========================================

export interface CreativePrediction {
    predicted_ctr: number;
    predicted_cvr: number;
    confidence_score: number;
    hook_strength_score: number;
    color_psychology_score: number;
    brand_consistency_score: number;
    text_sentiment_score: number;
    readability_score: number;
    overall_quality_score: number;
    recommendations: string[];
}

export const creativeApi = {
    predict: (data: {
        headline: string;
        body_text?: string;
        cta_type?: string;
        platform?: string;
    }): Promise<CreativePrediction> =>
        api.post('/creative/predict', data),

    predictWithUpload: (formData: FormData): Promise<CreativePrediction> =>
        api.upload('/creative/predict/upload', formData),

    getBenchmarks: (industry: string, platform?: string) =>
        api.get('/creative/benchmarks/' + industry, { platform }),
};

// ===========================================
// ROAS OPTIMIZER SERVICE
// ===========================================

export interface CampaignMetrics {
    campaign_id: string;
    campaign_name: string;
    current_budget: number;
    spend_today: number;
    impressions: number;
    clicks: number;
    conversions: number;
    revenue: number;
    ctr: number;
    cvr: number;
    roas: number;
}

export interface BudgetRecommendation {
    campaign_id: string;
    campaign_name: string;
    current_budget: number;
    recommended_budget: number;
    change_percentage: number;
    predicted_roas: number;
    confidence_score: number;
    reasoning: string;
    action: 'increase' | 'decrease' | 'pause' | 'maintain';
}

export interface OptimizationResult {
    timestamp: string;
    total_campaigns: number;
    campaigns_adjusted: number;
    recommendations: BudgetRecommendation[];
}

export const roasApi = {
    getCampaigns: (params?: { client_id?: string; platform?: string }): Promise<CampaignMetrics[]> =>
        api.get('/roas/campaigns', params),

    optimize: (params?: { dry_run?: boolean }): Promise<OptimizationResult> =>
        api.post('/roas/optimize' + (params?.dry_run ? '?dry_run=true' : ''), {}),

    getForecast: (campaignId: string) =>
        api.get(`/roas/forecast/${campaignId}`),

    pauseCampaign: (campaignId: string, reason?: string) =>
        api.post(`/roas/pause/${campaignId}`, { reason }),

    resumeCampaign: (campaignId: string) =>
        api.post(`/roas/resume/${campaignId}`),
};

// ===========================================
// CHURN PREDICTION SERVICE
// ===========================================

export interface HealthScore {
    overall_score: number;
    roas_score: number;
    engagement_score: number;
    payment_score: number;
    communication_score: number;
    growth_score: number;
}

export interface ChurnPrediction {
    churn_probability: number;
    risk_level: 'critical' | 'warning' | 'monitor' | 'healthy';
    days_to_churn?: number;
}

export interface RiskFactor {
    factor: string;
    severity: 'high' | 'medium' | 'low';
    impact_score: number;
    description: string;
    trend: 'improving' | 'stable' | 'declining';
}

export interface ClientHealthReport {
    client_id: string;
    client_name: string;
    health_score: HealthScore;
    churn_prediction: ChurnPrediction;
    risk_factors: RiskFactor[];
    recommended_actions: string[];
}

export interface ChurnAlert {
    id: string;
    client_id: string;
    client_name: string;
    alert_type: 'critical' | 'warning' | 'monitor';
    health_score: number;
    churn_probability: number;
    triggered_at: string;
    status: 'open' | 'acknowledged' | 'resolved';
}

export const churnApi = {
    getClientHealth: (clientId: string): Promise<ClientHealthReport> =>
        api.get(`/churn/health/${clientId}`),

    getAllHealth: (params?: { risk_level?: string }): Promise<ClientHealthReport[]> =>
        api.get('/churn/health', params),

    getAlerts: (params?: { status?: string }): Promise<ChurnAlert[]> =>
        api.get('/churn/alerts', params),

    acknowledgeAlert: (alertId: string) =>
        api.post(`/churn/alerts/${alertId}/acknowledge`),

    resolveAlert: (alertId: string, notes: string) =>
        api.post(`/churn/alerts/${alertId}/resolve`, { resolution_notes: notes }),

    getAtRisk: (threshold?: number) =>
        api.get('/churn/at-risk', { threshold }),
};

// ===========================================
// ATTRIBUTION SERVICE
// ===========================================

export interface ChannelAttribution {
    channel: string;
    shapley_attribution: number;
    markov_attribution: number;
    first_touch_attribution: number;
    last_touch_attribution: number;
    contribution_percentage: number;
}

export interface AttributionReport {
    client_id: string;
    total_conversions: number;
    total_value: number;
    channel_attributions: ChannelAttribution[];
    top_paths: Array<{ path: string[] | string; conversions: number }>;
    insights: string[];
}

export const attributionApi = {
    getReport: (clientId: string, startDate: string, endDate: string): Promise<AttributionReport> =>
        api.get(`/attribution/report/${clientId}`, { start_date: startDate, end_date: endDate }),

    getPathAnalysis: (clientId: string, startDate: string, endDate: string) =>
        api.get(`/attribution/paths/${clientId}`, { start_date: startDate, end_date: endDate }),

    getBudgetRecommendation: (clientId: string, totalBudget: number) =>
        api.get(`/attribution/budget-recommendation/${clientId}`, { total_budget: totalBudget }),
};

// ===========================================
// AUDIENCE SERVICE
// ===========================================

export interface AudienceSegment {
    id: string;
    name: string;
    segment_type: 'cluster' | 'lookalike' | 'custom';
    estimated_size: number;
    status: 'draft' | 'active' | 'synced';
    platforms_synced: string[];
}

export const audienceApi = {
    getSegments: (clientId: string): Promise<AudienceSegment[]> =>
        api.get(`/audience/segments/${clientId}`),

    createSegment: (clientId: string, data: { name: string; segment_type: string }) =>
        api.post('/audience/segments', { client_id: clientId, ...data }),

    runClustering: (clientId: string, nClusters?: number) =>
        api.post(`/audience/cluster/${clientId}` + (nClusters ? `?n_clusters=${nClusters}` : ''), {}),

    syncToPlatform: (segmentId: string, platform: string) =>
        api.post(`/audience/segments/${segmentId}/sync?platform=${platform}`, {}),

    getRecommendations: (clientId: string) =>
        api.get(`/audience/recommendations/${clientId}`),
};

// ===========================================
// CLIENTS & CAMPAIGNS SERVICE
// ===========================================

export interface Client {
    id: string;
    name: string;
    industry?: string;
    monthly_budget?: number;
    health_score: number;
    churn_probability?: number;
    is_active: boolean;
}

export interface Campaign {
    id: string;
    name: string;
    platform: string;
    status: string;
    daily_budget?: number;
    predicted_roas?: number;
}

export const clientsApi = {
    list: (params?: object): Promise<PaginatedResponse<Client>> =>
        api.get('/clients', params),

    get: (clientId: string): Promise<Client> =>
        api.get(`/clients/${clientId}`),

    create: (data: Partial<Client>): Promise<Client> =>
        api.post('/clients', data),

    update: (clientId: string, data: Partial<Client>): Promise<Client> =>
        api.put(`/clients/${clientId}`, data),
};

export const campaignsApi = {
    list: (params?: object): Promise<PaginatedResponse<Campaign>> =>
        api.get('/campaigns', params),

    get: (campaignId: string): Promise<Campaign> =>
        api.get(`/campaigns/${campaignId}`),

    sync: (adAccountId: string, fullSync?: boolean) =>
        api.post('/campaigns/sync', { ad_account_id: adAccountId, full_sync: fullSync }),
};

// ===========================================
// DASHBOARD SUMMARY SERVICE
// ===========================================

export interface DashboardStat {
    label: string;
    value: string;
    change: string;
    change_type: 'positive' | 'negative' | 'neutral';
}

export interface TopCampaign {
    name: string;
    roas: number;
    spend: number;
    status: string;
}

export interface DashboardAlert {
    type: 'warning' | 'success' | 'info';
    message: string;
    time: string;
}

export interface DashboardSummary {
    stats: DashboardStat[];
    top_campaigns: TopCampaign[];
    recent_alerts: DashboardAlert[];
    model_accuracy: Record<string, number>;
}

export interface ModelMetrics {
    creative_predictor: {
        accuracy: number;
        precision: number;
        recall: number;
        f1_score: number;
        predictions_made: number;
        last_trained: string;
    };
    roas_optimizer: {
        accuracy: number;
        avg_roas_improvement: number;
        optimizations_run: number;
        campaigns_optimized: number;
    };
    churn_predictor: {
        accuracy: number;
        early_detection_days: number;
        true_positive_rate: number;
        clients_monitored: number;
    };
    attribution_engine: {
        shapley_confidence: number;
        markov_accuracy: number;
        journeys_analyzed: number;
    };
    audience_segmenter: {
        silhouette_score: number;
        segments_created: number;
        users_segmented: number;
    };
}

export const dashboardApi = {
    getSummary: (): Promise<DashboardSummary> =>
        api.get('/dashboard/summary'),

    getModelMetrics: (): Promise<ModelMetrics> =>
        api.get('/dashboard/model-metrics'),
};

export default api;

