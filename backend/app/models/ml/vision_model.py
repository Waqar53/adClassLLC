"""
Computer Vision Models for Creative Analysis

ResNet, EfficientNet for image analysis and video hook detection.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path


class ImageFeatureType(str, Enum):
    COLOR = "color"
    COMPOSITION = "composition"
    OBJECTS = "objects"
    TEXT = "text"
    FACES = "faces"
    BRAND = "brand"


@dataclass
class ColorAnalysis:
    """Color analysis results."""
    dominant_colors: List[Tuple[int, int, int]]  # RGB values
    color_harmony_score: float  # 0-1
    brightness: float  # 0-1
    saturation: float  # 0-1
    contrast: float  # 0-1
    psychology_scores: Dict[str, float]  # emotion associations


@dataclass
class CompositionAnalysis:
    """Image composition analysis."""
    rule_of_thirds_score: float
    symmetry_score: float
    balance_score: float
    focal_point: Optional[Tuple[float, float]]
    negative_space_ratio: float


@dataclass
class ObjectDetection:
    """Detected objects in image."""
    objects: List[Dict[str, Any]]  # {class, confidence, bbox}
    scene_type: str
    product_visible: bool
    text_regions: List[Dict[str, Any]]


@dataclass
class FaceAnalysis:
    """Face detection and analysis."""
    face_count: int
    faces: List[Dict[str, Any]]  # {bbox, emotion, age_range, gender}
    avg_emotion: str
    looking_at_camera: bool


@dataclass
class VideoAnalysis:
    """Video analysis results."""
    duration_seconds: float
    frame_count: int
    hook_score: float  # First 3 seconds engagement
    key_frames: List[Dict[str, Any]]
    scene_changes: List[float]  # Timestamps
    audio_features: Dict[str, Any]


class ImageFeatureExtractor:
    """
    Extract features from images using pre-trained CNN.
    
    In production: Uses ResNet50 or EfficientNet-B3.
    """
    
    def __init__(self, model_name: str = "efficientnet_b3"):
        self.model_name = model_name
        self.model = None
        self.feature_dim = 1536  # EfficientNet-B3 feature dimension
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model."""
        # In production:
        # import torch
        # import torchvision.models as models
        # self.model = models.efficientnet_b3(pretrained=True)
        # self.model.eval()
        pass
    
    def extract_features(self, image_data: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from image.
        
        Args:
            image_data: RGB image array (H, W, 3)
            
        Returns:
            Feature vector of shape (feature_dim,)
        """
        # In production: actual model inference
        # Mock feature extraction
        np.random.seed(hash(image_data.tobytes()) % 2**32)
        return np.random.randn(self.feature_dim).astype(np.float32)
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract features for batch of images."""
        return np.stack([self.extract_features(img) for img in images])


class ColorAnalyzer:
    """
    Analyze color properties of images.
    """
    
    # Color psychology mappings
    COLOR_EMOTIONS = {
        "red": {"excitement": 0.9, "urgency": 0.8, "passion": 0.85},
        "blue": {"trust": 0.9, "calm": 0.85, "professional": 0.8},
        "green": {"nature": 0.9, "health": 0.85, "growth": 0.8},
        "yellow": {"optimism": 0.85, "attention": 0.9, "warmth": 0.75},
        "orange": {"energy": 0.85, "fun": 0.8, "enthusiasm": 0.85},
        "purple": {"luxury": 0.85, "creativity": 0.8, "mystery": 0.75},
        "black": {"sophistication": 0.9, "power": 0.85, "elegance": 0.8},
        "white": {"purity": 0.85, "simplicity": 0.9, "clean": 0.85},
    }
    
    def analyze(self, image_data: np.ndarray) -> ColorAnalysis:
        """
        Analyze image colors.
        
        Args:
            image_data: RGB image array
        """
        # Mock analysis - in production would use actual color extraction
        height, width = image_data.shape[:2] if len(image_data.shape) >= 2 else (100, 100)
        
        # Simulate dominant colors extraction
        dominant_colors = [
            (66, 133, 244),   # Blue
            (255, 255, 255),  # White
            (52, 168, 83),    # Green
        ]
        
        # Calculate mock metrics
        brightness = 0.65
        saturation = 0.72
        contrast = 0.68
        
        # Color harmony (complementary, analogous, etc.)
        harmony_score = 0.78
        
        # Psychology scores based on dominant colors
        psychology_scores = {
            "trust": 0.85,
            "professional": 0.80,
            "calm": 0.75,
            "energy": 0.60,
        }
        
        return ColorAnalysis(
            dominant_colors=dominant_colors,
            color_harmony_score=harmony_score,
            brightness=brightness,
            saturation=saturation,
            contrast=contrast,
            psychology_scores=psychology_scores
        )
    
    def get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Get color name from RGB values."""
        r, g, b = rgb
        
        # Simple color classification
        if r > 200 and g < 100 and b < 100:
            return "red"
        elif r < 100 and g < 100 and b > 200:
            return "blue"
        elif r < 100 and g > 200 and b < 100:
            return "green"
        elif r > 200 and g > 200 and b < 100:
            return "yellow"
        elif r > 200 and g > 100 and b < 100:
            return "orange"
        elif r > 100 and g < 100 and b > 200:
            return "purple"
        elif r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        
        return "neutral"


class CompositionAnalyzer:
    """
    Analyze image composition.
    """
    
    def analyze(self, image_data: np.ndarray) -> CompositionAnalysis:
        """Analyze image composition."""
        height, width = image_data.shape[:2] if len(image_data.shape) >= 2 else (100, 100)
        
        # Mock analysis
        return CompositionAnalysis(
            rule_of_thirds_score=0.82,
            symmetry_score=0.65,
            balance_score=0.78,
            focal_point=(0.35, 0.40),  # Normalized coordinates
            negative_space_ratio=0.32
        )


class ObjectDetector:
    """
    Detect objects in images using YOLO or similar.
    """
    
    def __init__(self, model_name: str = "yolov8"):
        self.model_name = model_name
        self.model = None
        self.class_names = [
            "person", "product", "text", "logo", "button",
            "phone", "laptop", "car", "food", "clothing"
        ]
    
    def detect(self, image_data: np.ndarray) -> ObjectDetection:
        """Detect objects in image."""
        # Mock detection
        objects = [
            {"class": "product", "confidence": 0.92, "bbox": [0.2, 0.3, 0.8, 0.9]},
            {"class": "text", "confidence": 0.88, "bbox": [0.1, 0.05, 0.9, 0.15]},
        ]
        
        text_regions = [
            {"text": "50% OFF", "confidence": 0.95, "bbox": [0.1, 0.05, 0.5, 0.12]},
            {"text": "Shop Now", "confidence": 0.90, "bbox": [0.3, 0.85, 0.7, 0.95]},
        ]
        
        return ObjectDetection(
            objects=objects,
            scene_type="product_showcase",
            product_visible=True,
            text_regions=text_regions
        )


class FaceDetector:
    """
    Detect and analyze faces in images.
    """
    
    EMOTIONS = ["neutral", "happy", "surprised", "sad", "angry", "disgusted", "fearful"]
    
    def detect(self, image_data: np.ndarray) -> FaceAnalysis:
        """Detect and analyze faces."""
        # Mock detection
        faces = [
            {
                "bbox": [0.3, 0.2, 0.7, 0.6],
                "emotion": "happy",
                "emotion_confidence": 0.85,
                "age_range": (25, 35),
                "gender": "female",
                "gender_confidence": 0.92
            }
        ]
        
        return FaceAnalysis(
            face_count=len(faces),
            faces=faces,
            avg_emotion="happy",
            looking_at_camera=True
        )


class VideoAnalyzer:
    """
    Analyze video creatives.
    """
    
    def __init__(self):
        self.feature_extractor = ImageFeatureExtractor()
        self.color_analyzer = ColorAnalyzer()
    
    def analyze(self, video_path: str, sample_rate: int = 1) -> VideoAnalysis:
        """
        Analyze video creative.
        
        Args:
            video_path: Path to video file
            sample_rate: Frames per second to sample
        """
        # Mock video analysis
        duration = 15.0
        fps = 30
        frame_count = int(duration * fps)
        
        # Hook score (first 3 seconds engagement prediction)
        hook_score = 0.78
        
        # Key frames
        key_frames = [
            {"timestamp": 0.0, "type": "intro", "features": {}},
            {"timestamp": 3.0, "type": "product_reveal", "features": {}},
            {"timestamp": 8.0, "type": "benefits", "features": {}},
            {"timestamp": 13.0, "type": "cta", "features": {}},
        ]
        
        # Scene changes
        scene_changes = [0.0, 3.2, 7.8, 12.5]
        
        # Audio features
        audio_features = {
            "has_music": True,
            "has_voiceover": True,
            "avg_volume": 0.72,
            "tempo_bpm": 120,
        }
        
        return VideoAnalysis(
            duration_seconds=duration,
            frame_count=frame_count,
            hook_score=hook_score,
            key_frames=key_frames,
            scene_changes=scene_changes,
            audio_features=audio_features
        )
    
    def calculate_hook_score(self, first_frames: List[np.ndarray]) -> float:
        """
        Calculate hook score from first 3 seconds of video.
        
        Factors:
        - Visual movement/change
        - Color vibrancy
        - Face presence
        - Text visibility
        """
        if not first_frames:
            return 0.5
        
        scores = []
        
        for frame in first_frames[:90]:  # First 3 seconds at 30fps
            color = self.color_analyzer.analyze(frame)
            scores.append(color.brightness * 0.3 + color.contrast * 0.3 + color.saturation * 0.4)
        
        return float(np.mean(scores)) if scores else 0.5


class CreativeVisionService:
    """
    Main service for visual creative analysis.
    """
    
    def __init__(self):
        self.feature_extractor = ImageFeatureExtractor()
        self.color_analyzer = ColorAnalyzer()
        self.composition_analyzer = CompositionAnalyzer()
        self.object_detector = ObjectDetector()
        self.face_detector = FaceDetector()
        self.video_analyzer = VideoAnalyzer()
    
    def analyze_image(self, image_data: np.ndarray) -> Dict[str, Any]:
        """
        Complete image analysis.
        
        Returns all visual features for the creative.
        """
        # Extract deep features
        features = self.feature_extractor.extract_features(image_data)
        
        # Analyze various aspects
        colors = self.color_analyzer.analyze(image_data)
        composition = self.composition_analyzer.analyze(image_data)
        objects = self.object_detector.detect(image_data)
        faces = self.face_detector.detect(image_data)
        
        # Calculate overall visual score
        visual_score = self._calculate_visual_score(colors, composition, objects, faces)
        
        return {
            "feature_vector": features.tolist(),
            "colors": {
                "dominant": colors.dominant_colors,
                "harmony_score": colors.color_harmony_score,
                "brightness": colors.brightness,
                "saturation": colors.saturation,
                "contrast": colors.contrast,
                "psychology": colors.psychology_scores,
            },
            "composition": {
                "rule_of_thirds": composition.rule_of_thirds_score,
                "symmetry": composition.symmetry_score,
                "balance": composition.balance_score,
                "focal_point": composition.focal_point,
                "negative_space": composition.negative_space_ratio,
            },
            "objects": {
                "detected": objects.objects,
                "scene_type": objects.scene_type,
                "has_product": objects.product_visible,
                "text_regions": objects.text_regions,
            },
            "faces": {
                "count": faces.face_count,
                "details": faces.faces,
                "emotion": faces.avg_emotion,
                "eye_contact": faces.looking_at_camera,
            },
            "overall_visual_score": visual_score,
        }
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Complete video analysis."""
        analysis = self.video_analyzer.analyze(video_path)
        
        return {
            "duration": analysis.duration_seconds,
            "frame_count": analysis.frame_count,
            "hook_score": analysis.hook_score,
            "key_frames": analysis.key_frames,
            "scene_changes": analysis.scene_changes,
            "audio": analysis.audio_features,
        }
    
    def _calculate_visual_score(
        self,
        colors: ColorAnalysis,
        composition: CompositionAnalysis,
        objects: ObjectDetection,
        faces: FaceAnalysis
    ) -> float:
        """Calculate overall visual quality score."""
        scores = [
            colors.color_harmony_score * 0.2,
            colors.contrast * 0.1,
            composition.rule_of_thirds_score * 0.15,
            composition.balance_score * 0.15,
            1.0 if objects.product_visible else 0.5,
            0.2,  # Base score for text presence
            1.0 if faces.looking_at_camera else 0.8,
        ]
        
        weights = [0.2, 0.1, 0.15, 0.15, 0.15, 0.1, 0.15]
        
        return sum(s * w for s, w in zip(scores, weights))


# Singleton
_vision_service: Optional[CreativeVisionService] = None


def get_vision_service() -> CreativeVisionService:
    """Get or create vision service instance."""
    global _vision_service
    if _vision_service is None:
        _vision_service = CreativeVisionService()
    return _vision_service
