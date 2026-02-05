-- AdClass AI Platform Database Initialization
-- This script runs on first database initialization

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ===========================================
-- CLIENTS & ACCOUNTS
-- ===========================================

CREATE TABLE IF NOT EXISTS clients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    website VARCHAR(255),
    monthly_budget DECIMAL(15,2),
    contract_start_date DATE,
    contract_end_date DATE,
    health_score INTEGER DEFAULT 50 CHECK (health_score >= 0 AND health_score <= 100),
    churn_probability DECIMAL(5,4) CHECK (churn_probability >= 0 AND churn_probability <= 1),
    slack_channel_id VARCHAR(100),
    primary_contact_email VARCHAR(255),
    account_manager_id UUID,
    notes TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ad_accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID REFERENCES clients(id) ON DELETE CASCADE,
    platform VARCHAR(50) NOT NULL CHECK (platform IN ('meta', 'google', 'tiktok', 'linkedin', 'twitter')),
    account_id VARCHAR(100) NOT NULL,
    account_name VARCHAR(255),
    access_token_encrypted TEXT,
    refresh_token_encrypted TEXT,
    token_expires_at TIMESTAMPTZ,
    currency VARCHAR(3) DEFAULT 'USD',
    timezone VARCHAR(50) DEFAULT 'UTC',
    is_active BOOLEAN DEFAULT true,
    last_synced_at TIMESTAMPTZ,
    sync_status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(platform, account_id)
);

-- ===========================================
-- CAMPAIGNS & AD GROUPS
-- ===========================================

CREATE TABLE IF NOT EXISTS campaigns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ad_account_id UUID REFERENCES ad_accounts(id) ON DELETE CASCADE,
    platform_campaign_id VARCHAR(100) NOT NULL,
    name VARCHAR(255) NOT NULL,
    objective VARCHAR(100),
    status VARCHAR(50) DEFAULT 'active',
    buying_type VARCHAR(50),
    daily_budget DECIMAL(12,2),
    lifetime_budget DECIMAL(15,2),
    start_date DATE,
    end_date DATE,
    bid_strategy VARCHAR(100),
    optimization_goal VARCHAR(100),
    -- ML-derived fields
    predicted_roas DECIMAL(8,4),
    risk_score DECIMAL(4,2),
    optimization_status VARCHAR(50),
    last_optimized_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ad_account_id, platform_campaign_id)
);

CREATE TABLE IF NOT EXISTS ad_sets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    platform_adset_id VARCHAR(100) NOT NULL,
    name VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    daily_budget DECIMAL(12,2),
    lifetime_budget DECIMAL(15,2),
    targeting JSONB, -- Store targeting criteria
    placements JSONB,
    schedule JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(campaign_id, platform_adset_id)
);

-- ===========================================
-- AD CREATIVES
-- ===========================================

CREATE TABLE IF NOT EXISTS ad_creatives (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ad_set_id UUID REFERENCES ad_sets(id) ON DELETE CASCADE,
    platform_ad_id VARCHAR(100) NOT NULL,
    name VARCHAR(255),
    creative_type VARCHAR(50) CHECK (creative_type IN ('image', 'video', 'carousel', 'collection', 'dynamic')),
    status VARCHAR(50) DEFAULT 'active',
    
    -- Creative content
    headline TEXT,
    body_text TEXT,
    description TEXT,
    cta_type VARCHAR(50),
    destination_url TEXT,
    display_url VARCHAR(255),
    
    -- Media
    media_url TEXT,
    thumbnail_url TEXT,
    video_duration_seconds INTEGER,
    
    -- ML Predictions
    predicted_ctr DECIMAL(6,4),
    predicted_cvr DECIMAL(6,4),
    predicted_roas DECIMAL(8,4),
    hook_strength_score DECIMAL(4,2),
    color_psychology_score DECIMAL(4,2),
    brand_consistency_score DECIMAL(4,2),
    text_sentiment_score DECIMAL(4,2),
    readability_score DECIMAL(4,2),
    overall_quality_score DECIMAL(4,2),
    prediction_confidence DECIMAL(4,2),
    
    -- Extracted features (for ML)
    visual_features JSONB,
    text_features JSONB,
    
    -- Performance (aggregated)
    total_impressions BIGINT DEFAULT 0,
    total_clicks BIGINT DEFAULT 0,
    total_conversions BIGINT DEFAULT 0,
    total_spend DECIMAL(15,2) DEFAULT 0,
    total_revenue DECIMAL(15,2) DEFAULT 0,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ad_set_id, platform_ad_id)
);

-- ===========================================
-- PERFORMANCE METRICS (Time-series)
-- ===========================================

CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(50) NOT NULL CHECK (entity_type IN ('account', 'campaign', 'adset', 'ad')),
    entity_id UUID NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    
    -- Core metrics
    impressions INTEGER DEFAULT 0,
    reach INTEGER DEFAULT 0,
    clicks INTEGER DEFAULT 0,
    unique_clicks INTEGER DEFAULT 0,
    conversions INTEGER DEFAULT 0,
    conversion_value DECIMAL(12,4) DEFAULT 0,
    spend DECIMAL(12,4) DEFAULT 0,
    
    -- Calculated metrics
    ctr DECIMAL(8,6),
    cpc DECIMAL(10,4),
    cpm DECIMAL(10,4),
    cvr DECIMAL(8,6),
    cpa DECIMAL(10,4),
    roas DECIMAL(10,4),
    
    -- Engagement
    likes INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    shares INTEGER DEFAULT 0,
    video_views INTEGER DEFAULT 0,
    video_views_p25 INTEGER DEFAULT 0,
    video_views_p50 INTEGER DEFAULT 0,
    video_views_p75 INTEGER DEFAULT 0,
    video_views_p100 INTEGER DEFAULT 0,
    
    -- Additional data
    breakdown JSONB, -- Age, gender, placement, etc.
    
    PRIMARY KEY (id, recorded_at)
) PARTITION BY RANGE (recorded_at);

-- Create monthly partitions for the next 12 months
CREATE TABLE performance_metrics_2026_01 PARTITION OF performance_metrics
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE performance_metrics_2026_02 PARTITION OF performance_metrics
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE performance_metrics_2026_03 PARTITION OF performance_metrics
    FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE performance_metrics_2026_04 PARTITION OF performance_metrics
    FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE performance_metrics_2026_05 PARTITION OF performance_metrics
    FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE performance_metrics_2026_06 PARTITION OF performance_metrics
    FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');

-- ===========================================
-- ATTRIBUTION DATA
-- ===========================================

CREATE TABLE IF NOT EXISTS customer_journeys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID REFERENCES clients(id) ON DELETE CASCADE,
    customer_id VARCHAR(255), -- Hashed customer identifier
    conversion_id VARCHAR(100),
    conversion_type VARCHAR(100),
    conversion_value DECIMAL(12,2),
    converted_at TIMESTAMPTZ,
    first_touch_at TIMESTAMPTZ,
    journey_length INTEGER, -- Number of touchpoints
    journey_duration_hours INTEGER,
    journey_data JSONB, -- Full touchpoint array
    calculated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS touchpoint_attributions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    journey_id UUID REFERENCES customer_journeys(id) ON DELETE CASCADE,
    touchpoint_order INTEGER NOT NULL,
    channel VARCHAR(100) NOT NULL,
    campaign_id UUID REFERENCES campaigns(id),
    interaction_type VARCHAR(50), -- click, view, engagement
    occurred_at TIMESTAMPTZ NOT NULL,
    
    -- Attribution values
    shapley_value DECIMAL(8,4),
    markov_value DECIMAL(8,4),
    first_touch_value DECIMAL(8,4),
    last_touch_value DECIMAL(8,4),
    linear_value DECIMAL(8,4),
    time_decay_value DECIMAL(8,4),
    position_based_value DECIMAL(8,4),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ===========================================
-- CLIENT HEALTH & CHURN
-- ===========================================

CREATE TABLE IF NOT EXISTS client_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID REFERENCES clients(id) ON DELETE CASCADE,
    interaction_type VARCHAR(50) NOT NULL CHECK (
        interaction_type IN ('slack_message', 'email', 'call', 'meeting', 
                            'dashboard_login', 'support_ticket', 'payment')
    ),
    direction VARCHAR(20), -- inbound, outbound
    occurred_at TIMESTAMPTZ NOT NULL,
    response_time_minutes INTEGER,
    sentiment_score DECIMAL(4,2),
    subject TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS client_health_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID REFERENCES clients(id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL,
    
    -- Health components
    overall_score INTEGER CHECK (overall_score >= 0 AND overall_score <= 100),
    roas_score INTEGER,
    engagement_score INTEGER,
    payment_score INTEGER,
    communication_score INTEGER,
    growth_score INTEGER,
    
    -- Risk metrics
    churn_probability DECIMAL(5,4),
    days_to_predicted_churn INTEGER,
    risk_factors JSONB,
    
    -- Raw metrics
    current_roas DECIMAL(8,4),
    target_roas DECIMAL(8,4),
    monthly_spend DECIMAL(15,2),
    dashboard_logins_30d INTEGER,
    avg_response_time_hours DECIMAL(6,2),
    tickets_open INTEGER,
    revenue_mom_change DECIMAL(6,2),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(client_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS churn_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID REFERENCES clients(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) CHECK (alert_type IN ('critical', 'warning', 'monitor')),
    triggered_at TIMESTAMPTZ DEFAULT NOW(),
    health_score_at_trigger INTEGER,
    churn_probability_at_trigger DECIMAL(5,4),
    risk_factors JSONB,
    recommended_actions JSONB,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by UUID,
    resolution_status VARCHAR(50) DEFAULT 'open',
    resolution_notes TEXT,
    resolved_at TIMESTAMPTZ
);

-- ===========================================
-- AUDIENCE SEGMENTS
-- ===========================================

CREATE TABLE IF NOT EXISTS audience_segments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID REFERENCES clients(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    segment_type VARCHAR(50) CHECK (segment_type IN ('cluster', 'lookalike', 'custom', 'retargeting')),
    status VARCHAR(50) DEFAULT 'draft',
    
    -- Size and coverage
    estimated_size INTEGER,
    actual_size INTEGER,
    match_rate DECIMAL(5,2),
    
    -- Platform sync
    platform VARCHAR(50),
    platform_audience_id VARCHAR(100),
    last_synced_at TIMESTAMPTZ,
    
    -- ML Features
    clustering_features JSONB,
    segment_characteristics JSONB,
    
    -- Performance
    avg_ctr DECIMAL(6,4),
    avg_cvr DECIMAL(6,4),
    avg_roas DECIMAL(8,4),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ===========================================
-- ML MODEL REGISTRY
-- ===========================================

CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(100), -- creative_predictor, roas_optimizer, etc.
    framework VARCHAR(50), -- pytorch, sklearn, xgboost
    
    -- Storage
    artifact_path TEXT NOT NULL,
    file_size_bytes BIGINT,
    
    -- Metrics
    training_metrics JSONB,
    validation_metrics JSONB,
    
    -- Status
    status VARCHAR(50) DEFAULT 'training',
    is_production BOOLEAN DEFAULT false,
    
    -- Metadata
    hyperparameters JSONB,
    feature_names JSONB,
    training_data_info JSONB,
    
    trained_at TIMESTAMPTZ,
    deployed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(model_name, model_version)
);

-- ===========================================
-- OPTIMIZATION HISTORY
-- ===========================================

CREATE TABLE IF NOT EXISTS optimization_actions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    action_type VARCHAR(50) CHECK (
        action_type IN ('budget_increase', 'budget_decrease', 'pause', 
                       'resume', 'bid_adjustment', 'targeting_change')
    ),
    executed_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Before/After values
    previous_value JSONB,
    new_value JSONB,
    
    -- Reasoning
    model_prediction JSONB,
    confidence_score DECIMAL(4,2),
    trigger_reason TEXT,
    
    -- Results
    was_successful BOOLEAN,
    result_metrics JSONB,
    rollback_at TIMESTAMPTZ
);

-- ===========================================
-- INDEXES
-- ===========================================

-- Clients
CREATE INDEX idx_clients_health ON clients(health_score);
CREATE INDEX idx_clients_churn ON clients(churn_probability);
CREATE INDEX idx_clients_active ON clients(is_active);

-- Ad Accounts
CREATE INDEX idx_ad_accounts_client ON ad_accounts(client_id);
CREATE INDEX idx_ad_accounts_platform ON ad_accounts(platform);

-- Campaigns
CREATE INDEX idx_campaigns_account ON campaigns(ad_account_id);
CREATE INDEX idx_campaigns_status ON campaigns(status);
CREATE INDEX idx_campaigns_dates ON campaigns(start_date, end_date);

-- Ad Creatives
CREATE INDEX idx_creatives_adset ON ad_creatives(ad_set_id);
CREATE INDEX idx_creatives_predicted_ctr ON ad_creatives(predicted_ctr DESC);
CREATE INDEX idx_creatives_type ON ad_creatives(creative_type);

-- Performance Metrics
CREATE INDEX idx_metrics_entity ON performance_metrics(entity_type, entity_id, recorded_at DESC);
CREATE INDEX idx_metrics_recorded ON performance_metrics(recorded_at DESC);

-- Customer Journeys
CREATE INDEX idx_journeys_client ON customer_journeys(client_id);
CREATE INDEX idx_journeys_converted ON customer_journeys(converted_at);

-- Client Interactions
CREATE INDEX idx_interactions_client ON client_interactions(client_id, occurred_at DESC);
CREATE INDEX idx_interactions_type ON client_interactions(interaction_type);

-- Audience Segments
CREATE INDEX idx_segments_client ON audience_segments(client_id);
CREATE INDEX idx_segments_type ON audience_segments(segment_type);

-- Full text search on creatives
CREATE INDEX idx_creatives_headline_trgm ON ad_creatives USING gin(headline gin_trgm_ops);
CREATE INDEX idx_creatives_body_trgm ON ad_creatives USING gin(body_text gin_trgm_ops);

-- ===========================================
-- FUNCTIONS
-- ===========================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to tables with updated_at
CREATE TRIGGER update_clients_updated_at BEFORE UPDATE ON clients
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_ad_accounts_updated_at BEFORE UPDATE ON ad_accounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_campaigns_updated_at BEFORE UPDATE ON campaigns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_ad_sets_updated_at BEFORE UPDATE ON ad_sets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_ad_creatives_updated_at BEFORE UPDATE ON ad_creatives
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_audience_segments_updated_at BEFORE UPDATE ON audience_segments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Log initial setup
DO $$
BEGIN
    RAISE NOTICE 'AdClass AI Platform database initialized successfully!';
END $$;
