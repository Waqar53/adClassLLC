"""
ML Models Package

Contains production ML models for all 5 AI modules:
1. Creative Predictor - XGBoost for CTR/CVR prediction
2. ROAS Optimizer - Thompson Sampling + LSTM forecasting
3. Churn Predictor - Random Forest for churn probability
4. Attribution Engine - Shapley values + Markov chains
5. Audience Segmenter - K-Means clustering + RFM analysis
"""

from .creative_predictor import CreativePredictor, get_creative_predictor
from .roas_optimizer import ROASOptimizer, CampaignState, get_roas_optimizer
from .churn_predictor import ChurnPredictor, ClientMetrics, get_churn_predictor
from .attribution_engine import AttributionEngine, ConversionPath, Touchpoint, get_attribution_engine
from .audience_segmenter import AudienceSegmenter, UserProfile, get_audience_segmenter

__all__ = [
    # Creative Predictor
    "CreativePredictor",
    "get_creative_predictor",
    
    # ROAS Optimizer
    "ROASOptimizer",
    "CampaignState",
    "get_roas_optimizer",
    
    # Churn Predictor
    "ChurnPredictor",
    "ClientMetrics",
    "get_churn_predictor",
    
    # Attribution Engine
    "AttributionEngine",
    "ConversionPath",
    "Touchpoint",
    "get_attribution_engine",
    
    # Audience Segmenter
    "AudienceSegmenter",
    "UserProfile",
    "get_audience_segmenter",
]
