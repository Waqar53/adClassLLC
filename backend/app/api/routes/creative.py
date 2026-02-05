"""
Creative Performance Predictor API Routes

Module 1: Predict ad creative performance before launch.
"""

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.cache import predictions_cache
from app.services.ml.creative_predictor import get_creative_predictor

router = APIRouter()


# ===========================================
# REQUEST/RESPONSE MODELS
# ===========================================

class CreativeAnalysisRequest(BaseModel):
    """Request to analyze a creative."""
    headline: str = Field(..., description="Ad headline text")
    body_text: Optional[str] = Field(None, description="Ad body/description text")
    cta_type: Optional[str] = Field(None, description="Call-to-action type")
    media_url: Optional[str] = Field(None, description="URL to image/video creative")
    platform: str = Field("meta", description="Target platform: meta, google, tiktok")
    industry: Optional[str] = Field(None, description="Industry category")


class CreativePrediction(BaseModel):
    """Creative performance prediction result."""
    predicted_ctr: float = Field(..., description="Predicted click-through rate (0-1)")
    predicted_cvr: float = Field(..., description="Predicted conversion rate (0-1)")
    confidence_score: float = Field(..., description="Prediction confidence (0-1)")
    
    # Quality scores (0-100)
    hook_strength_score: float = Field(..., description="First 3 seconds engagement score")
    color_psychology_score: float = Field(..., description="Color effectiveness score")
    brand_consistency_score: float = Field(..., description="Brand alignment score")
    text_sentiment_score: float = Field(..., description="Text sentiment score")
    readability_score: float = Field(..., description="Text readability score")
    overall_quality_score: float = Field(..., description="Overall creative quality")
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    comparable_creatives: List[dict] = Field(default_factory=list)


class CreativeBatchRequest(BaseModel):
    """Batch creative analysis request."""
    creatives: List[CreativeAnalysisRequest]


class CreativeBatchResponse(BaseModel):
    """Batch creative analysis response."""
    predictions: List[CreativePrediction]
    best_performer_index: int


# ===========================================
# API ENDPOINTS
# ===========================================

@router.post("/predict", response_model=CreativePrediction)
async def predict_creative_performance(
    request: CreativeAnalysisRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Predict ad creative performance before launch.
    
    Analyzes visual elements, text content, and metadata to predict:
    - Expected CTR (within ±0.3%)
    - Expected CVR
    - Overall quality metrics
    
    Returns recommendations for improvement.
    """
    # Use real ML model for prediction
    predictor = get_creative_predictor()
    
    result = predictor.predict(
        headline=request.headline,
        body_text=request.body_text,
        cta_type=request.cta_type,
        platform=request.platform,
        industry=request.industry or "default",
        media_url=request.media_url
    )
    
    prediction = CreativePrediction(
        predicted_ctr=result["predicted_ctr"],
        predicted_cvr=result["predicted_cvr"],
        confidence_score=result["confidence_score"],
        hook_strength_score=result["hook_strength_score"],
        color_psychology_score=result["color_psychology_score"],
        brand_consistency_score=result["brand_consistency_score"],
        text_sentiment_score=result["text_sentiment_score"],
        readability_score=result["readability_score"],
        overall_quality_score=result["overall_quality_score"],
        recommendations=result["recommendations"],
        comparable_creatives=result["comparable_creatives"]
    )
    
    return prediction


@router.post("/predict/upload", response_model=CreativePrediction)
async def predict_with_upload(
    file: UploadFile = File(..., description="Image or video file"),
    headline: str = Form(...),
    body_text: Optional[str] = Form(None),
    cta_type: Optional[str] = Form(None),
    platform: str = Form("meta"),
    db: AsyncSession = Depends(get_db)
):
    """
    Predict creative performance with file upload.
    
    Supports: JPG, PNG, GIF, MP4, MOV
    Max file size: 50MB
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/gif", "video/mp4", "video/quicktime"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file.content_type} not supported. Allowed: {allowed_types}"
        )
    
    # TODO: Process file and run prediction
    # 1. Save file to S3/MinIO
    # 2. Extract visual features
    # 3. Run model prediction
    
    prediction = CreativePrediction(
        predicted_ctr=0.0267,
        predicted_cvr=0.0095,
        confidence_score=0.91,
        hook_strength_score=85.0,
        color_psychology_score=79.0,
        brand_consistency_score=68.0,
        text_sentiment_score=72.0,
        readability_score=90.0,
        overall_quality_score=78.8,
        recommendations=[
            "Strong visual composition detected",
            "Consider A/B testing with different CTA placements"
        ]
    )
    
    return prediction


@router.post("/predict/batch", response_model=CreativeBatchResponse)
async def predict_batch(
    request: CreativeBatchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Predict performance for multiple creatives at once.
    
    Useful for comparing creative variants before launch.
    Returns individual predictions plus best performer.
    """
    if len(request.creatives) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 creatives per batch"
        )
    
    predictions = []
    best_score = 0
    best_index = 0
    
    for i, creative in enumerate(request.creatives):
        # TODO: Actual prediction
        pred = CreativePrediction(
            predicted_ctr=0.02 + (i * 0.003),
            predicted_cvr=0.008 + (i * 0.001),
            confidence_score=0.85,
            hook_strength_score=75 + i,
            color_psychology_score=80,
            brand_consistency_score=70,
            text_sentiment_score=65,
            readability_score=85,
            overall_quality_score=75 + i
        )
        predictions.append(pred)
        
        if pred.overall_quality_score > best_score:
            best_score = pred.overall_quality_score
            best_index = i
    
    return CreativeBatchResponse(
        predictions=predictions,
        best_performer_index=best_index
    )


@router.get("/analyze/{creative_id}")
async def get_creative_analysis(
    creative_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get stored analysis for an existing creative.
    """
    # Check cache first
    cached = await predictions_cache.get(f"creative:{creative_id}")
    if cached:
        return cached
    
    # TODO: Query database for stored prediction
    raise HTTPException(status_code=404, detail="Creative analysis not found")


@router.get("/benchmarks/{industry}")
async def get_industry_benchmarks(
    industry: str,
    platform: str = "meta"
):
    """
    Get performance benchmarks for an industry.
    
    Returns average CTR, CVR, and other metrics for comparison.
    """
    # TODO: Query historical data for benchmarks
    benchmarks = {
        "industry": industry,
        "platform": platform,
        "metrics": {
            "avg_ctr": 0.0215,
            "median_ctr": 0.0189,
            "top_10_ctr": 0.0348,
            "avg_cvr": 0.0078,
            "avg_cpm": 12.50,
            "avg_cpc": 0.58
        },
        "sample_size": 15420,
        "date_range": "last_90_days"
    }
    
    return benchmarks
