/**
 * Custom Hooks for Data Fetching
 * 
 * Uses SWR for efficient data fetching with caching.
 */

import useSWR, { SWRConfiguration } from 'swr';
import {
    creativeApi,
    roasApi,
    churnApi,
    attributionApi,
    audienceApi,
    clientsApi,
    campaignsApi,
    CampaignMetrics,
    ClientHealthReport,
    ChurnAlert,
    AudienceSegment,
    Client,
    Campaign,
    PaginatedResponse,
} from '@/services/api';

// Default SWR config
const defaultConfig: SWRConfiguration = {
    revalidateOnFocus: false,
    revalidateOnReconnect: true,
    dedupingInterval: 5000,
};

// ===========================================
// ROAS OPTIMIZER HOOKS
// ===========================================

export function useCampaignMetrics(clientId?: string, platform?: string) {
    return useSWR<CampaignMetrics[]>(
        ['campaigns', clientId, platform],
        () => roasApi.getCampaigns({ client_id: clientId, platform }),
        {
            ...defaultConfig,
            refreshInterval: 60000, // Refresh every minute
        }
    );
}

export function useROASForecast(campaignId: string) {
    return useSWR(
        campaignId ? ['forecast', campaignId] : null,
        () => roasApi.getForecast(campaignId),
        defaultConfig
    );
}

// ===========================================
// CHURN PREDICTION HOOKS
// ===========================================

export function useClientHealth(clientId: string) {
    return useSWR<ClientHealthReport>(
        clientId ? ['health', clientId] : null,
        () => churnApi.getClientHealth(clientId),
        defaultConfig
    );
}

export function useAllClientHealth(riskLevel?: string) {
    return useSWR<ClientHealthReport[]>(
        ['health', 'all', riskLevel],
        () => churnApi.getAllHealth({ risk_level: riskLevel }),
        defaultConfig
    );
}

export function useChurnAlerts(status?: string) {
    return useSWR<ChurnAlert[]>(
        ['alerts', status],
        () => churnApi.getAlerts({ status }),
        {
            ...defaultConfig,
            refreshInterval: 30000, // Refresh every 30 seconds
        }
    );
}

export function useAtRiskClients(threshold?: number) {
    return useSWR(
        ['at-risk', threshold],
        () => churnApi.getAtRisk(threshold),
        defaultConfig
    );
}

// ===========================================
// ATTRIBUTION HOOKS
// ===========================================

export function useAttributionReport(
    clientId: string,
    startDate: string,
    endDate: string
) {
    return useSWR(
        clientId && startDate && endDate
            ? ['attribution', clientId, startDate, endDate]
            : null,
        () => attributionApi.getReport(clientId, startDate, endDate),
        defaultConfig
    );
}

export function usePathAnalysis(
    clientId: string,
    startDate: string,
    endDate: string
) {
    return useSWR(
        clientId && startDate && endDate
            ? ['paths', clientId, startDate, endDate]
            : null,
        () => attributionApi.getPathAnalysis(clientId, startDate, endDate),
        defaultConfig
    );
}

// ===========================================
// AUDIENCE HOOKS
// ===========================================

export function useAudienceSegments(clientId: string) {
    return useSWR<AudienceSegment[]>(
        clientId ? ['segments', clientId] : null,
        () => audienceApi.getSegments(clientId),
        defaultConfig
    );
}

export function useAudienceRecommendations(clientId: string) {
    return useSWR(
        clientId ? ['audience-recommendations', clientId] : null,
        () => audienceApi.getRecommendations(clientId),
        defaultConfig
    );
}

// ===========================================
// CLIENT & CAMPAIGN HOOKS
// ===========================================

export function useClients(params?: object) {
    return useSWR<PaginatedResponse<Client>>(
        ['clients', params],
        () => clientsApi.list(params),
        defaultConfig
    );
}

export function useClient(clientId: string) {
    return useSWR<Client>(
        clientId ? ['client', clientId] : null,
        () => clientsApi.get(clientId),
        defaultConfig
    );
}

export function useCampaigns(params?: object) {
    return useSWR<PaginatedResponse<Campaign>>(
        ['campaigns-list', params],
        () => campaignsApi.list(params),
        defaultConfig
    );
}

export function useCampaign(campaignId: string) {
    return useSWR<Campaign>(
        campaignId ? ['campaign', campaignId] : null,
        () => campaignsApi.get(campaignId),
        defaultConfig
    );
}

// ===========================================
// CREATIVE PREDICTOR HOOKS
// ===========================================

export function useIndustryBenchmarks(industry: string, platform?: string) {
    return useSWR(
        industry ? ['benchmarks', industry, platform] : null,
        () => creativeApi.getBenchmarks(industry, platform),
        {
            ...defaultConfig,
            revalidateOnFocus: false,
            dedupingInterval: 300000, // Cache for 5 minutes
        }
    );
}
