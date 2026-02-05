"""
Creative Performance Predictor ML Model

Multi-modal fusion network for predicting ad creative performance.
Combines visual (EfficientNet), text (BERT), and metadata features.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class CreativeInput:
    """Input data for creative prediction."""
    image: Optional[np.ndarray] = None  # RGB image array
    video_frames: Optional[List[np.ndarray]] = None  # First 3 seconds frames
    headline: str = ""
    body_text: str = ""
    cta_type: str = ""
    platform: str = "meta"
    industry: str = ""


@dataclass
class CreativeOutput:
    """Output from creative prediction model."""
    predicted_ctr: float
    predicted_cvr: float
    hook_strength_score: float
    color_psychology_score: float
    brand_consistency_score: float
    text_sentiment_score: float
    readability_score: float
    overall_quality_score: float
    confidence: float
    embeddings: np.ndarray  # For similarity search


class VisualEncoder(nn.Module):
    """
    Visual encoder using EfficientNet-B4.
    Extracts visual features from ad images/videos.
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load EfficientNet-B4 backbone
        # In production: from torchvision.models import efficientnet_b4
        self.backbone = nn.Sequential(
            # Simulated EfficientNet structure
            nn.Conv2d(3, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Projection to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(96, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (B, C, H, W)
        Returns:
            Visual embedding (B, 256)
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.projection(features)


class TextEncoder(nn.Module):
    """
    Text encoder using BERT-style architecture.
    Encodes headlines and body text.
    """
    
    def __init__(self, vocab_size: int = 30522, hidden_size: int = 768):
        super().__init__()
        
        # Simplified text encoder (use HuggingFace BERT in production)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(512, hidden_size)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=3072,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs (B, seq_len)
        Returns:
            Text embedding (B, 256)
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        x = self.embedding(input_ids) + self.position_embedding(positions)
        x = self.transformer(x)
        
        # Pool to single vector (CLS token style)
        x = x[:, 0, :]  # Use first token
        return self.projection(x)


class MetadataEncoder(nn.Module):
    """
    Encodes categorical and numerical metadata features.
    """
    
    def __init__(
        self,
        num_platforms: int = 5,
        num_industries: int = 50,
        num_cta_types: int = 20
    ):
        super().__init__()
        
        # Embeddings for categorical features
        self.platform_embedding = nn.Embedding(num_platforms, 16)
        self.industry_embedding = nn.Embedding(num_industries, 32)
        self.cta_embedding = nn.Embedding(num_cta_types, 16)
        
        # Combine embeddings
        self.projection = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
    def forward(
        self,
        platform_id: torch.Tensor,
        industry_id: torch.Tensor,
        cta_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            platform_id: Platform index (B,)
            industry_id: Industry index (B,)
            cta_id: CTA type index (B,)
        Returns:
            Metadata embedding (B, 64)
        """
        p = self.platform_embedding(platform_id)
        i = self.industry_embedding(industry_id)
        c = self.cta_embedding(cta_id)
        
        combined = torch.cat([p, i, c], dim=-1)
        return self.projection(combined)


class CreativePerformancePredictor(nn.Module):
    """
    Main multi-modal fusion model for creative performance prediction.
    
    Architecture:
    - Visual encoder (EfficientNet-B4) -> 256d
    - Text encoder (BERT) -> 256d
    - Metadata encoder -> 64d
    - Fusion layer -> 512d
    - Task-specific heads for CTR, CVR, quality scores
    """
    
    def __init__(self):
        super().__init__()
        
        # Encoders
        self.visual_encoder = VisualEncoder()
        self.text_encoder = TextEncoder()
        self.metadata_encoder = MetadataEncoder()
        
        # Fusion layer (256 + 256 + 64 = 576)
        self.fusion = nn.Sequential(
            nn.Linear(576, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task-specific output heads
        self.ctr_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.cvr_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.hook_strength_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.quality_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5),  # 5 quality scores
            nn.Sigmoid()
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        image: torch.Tensor,
        text_ids: torch.Tensor,
        platform_id: torch.Tensor,
        industry_id: torch.Tensor,
        cta_id: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Returns:
            Dictionary with all predictions
        """
        # Encode each modality
        visual_emb = self.visual_encoder(image)
        text_emb = self.text_encoder(text_ids)
        meta_emb = self.metadata_encoder(platform_id, industry_id, cta_id)
        
        # Fuse embeddings
        combined = torch.cat([visual_emb, text_emb, meta_emb], dim=-1)
        fused = self.fusion(combined)
        
        # Generate predictions
        ctr = self.ctr_head(fused)
        cvr = self.cvr_head(fused)
        hook = self.hook_strength_head(fused)
        quality = self.quality_head(fused)
        confidence = self.confidence_head(fused)
        
        return {
            'predicted_ctr': ctr.squeeze(-1),
            'predicted_cvr': cvr.squeeze(-1),
            'hook_strength': hook.squeeze(-1),
            'color_score': quality[:, 0],
            'brand_score': quality[:, 1],
            'sentiment_score': quality[:, 2],
            'readability_score': quality[:, 3],
            'overall_score': quality[:, 4],
            'confidence': confidence.squeeze(-1),
            'embeddings': fused  # For similarity search
        }


class CreativePredictorService:
    """
    High-level service for creative performance prediction.
    Handles preprocessing, inference, and postprocessing.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = CreativePerformancePredictor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, path: str):
        """Load trained model weights."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize to 380x380 (EfficientNet-B4 input size)
        # Normalize with ImageNet stats
        # In production, use torchvision transforms
        
        # Placeholder: assume image is already preprocessed
        tensor = torch.from_numpy(image).float()
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)
    
    def tokenize_text(self, headline: str, body: str) -> torch.Tensor:
        """Tokenize text for model input."""
        # In production, use HuggingFace tokenizer
        # Placeholder: random tokens
        tokens = torch.randint(0, 30000, (1, 128))
        return tokens.to(self.device)
    
    def predict(self, creative: CreativeInput) -> CreativeOutput:
        """
        Run prediction on a creative.
        
        Args:
            creative: Input creative data
            
        Returns:
            Prediction results
        """
        with torch.no_grad():
            # Preprocess inputs
            if creative.image is not None:
                image_tensor = self.preprocess_image(creative.image)
            else:
                # Default blank image
                image_tensor = torch.zeros(1, 3, 380, 380).to(self.device)
            
            text_tensor = self.tokenize_text(creative.headline, creative.body_text)
            
            # Encode categorical features
            platform_map = {'meta': 0, 'google': 1, 'tiktok': 2, 'linkedin': 3, 'twitter': 4}
            platform_id = torch.tensor([platform_map.get(creative.platform, 0)]).to(self.device)
            industry_id = torch.tensor([0]).to(self.device)  # Simplified
            cta_id = torch.tensor([0]).to(self.device)  # Simplified
            
            # Run model
            outputs = self.model(
                image_tensor,
                text_tensor,
                platform_id,
                industry_id,
                cta_id
            )
            
            # Convert to output format
            return CreativeOutput(
                predicted_ctr=outputs['predicted_ctr'].item(),
                predicted_cvr=outputs['predicted_cvr'].item(),
                hook_strength_score=outputs['hook_strength'].item() * 100,
                color_psychology_score=outputs['color_score'].item() * 100,
                brand_consistency_score=outputs['brand_score'].item() * 100,
                text_sentiment_score=outputs['sentiment_score'].item() * 100,
                readability_score=outputs['readability_score'].item() * 100,
                overall_quality_score=outputs['overall_score'].item() * 100,
                confidence=outputs['confidence'].item(),
                embeddings=outputs['embeddings'].cpu().numpy()
            )


# Singleton service instance
_predictor_service: Optional[CreativePredictorService] = None


def get_creative_predictor() -> CreativePredictorService:
    """Get or create the creative predictor service."""
    global _predictor_service
    if _predictor_service is None:
        _predictor_service = CreativePredictorService()
    return _predictor_service
