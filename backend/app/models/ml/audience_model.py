"""
Audience Intelligence ML Models

K-Means clustering, lookalike generation, and audience segmentation.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import defaultdict


@dataclass
class AudienceMember:
    """A single audience member/customer."""
    customer_id: str
    features: Dict[str, float]
    segment_id: Optional[str] = None
    lookalike_score: Optional[float] = None


@dataclass
class AudienceSegment:
    """An audience segment/cluster."""
    segment_id: str
    name: str
    size: int
    centroid: Dict[str, float]
    characteristics: List[str]
    avg_value: float
    platforms_synced: List[str]


class KMeansClusterer:
    """
    K-Means clustering for audience segmentation.
    
    Groups customers based on behavioral and demographic features.
    """
    
    def __init__(self, n_clusters: int = 5, max_iterations: int = 100):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
    
    def fit(self, members: List[AudienceMember]) -> List[int]:
        """
        Fit K-Means clustering on audience members.
        
        Args:
            members: List of audience members with features
            
        Returns:
            Cluster assignments for each member
        """
        if not members:
            return []
        
        # Extract features
        self.feature_names = list(members[0].features.keys())
        X = np.array([
            [m.features.get(f, 0) for f in self.feature_names]
            for m in members
        ])
        
        # Normalize features
        X_normalized = self._normalize(X)
        
        # Initialize centroids randomly
        n_samples = X_normalized.shape[0]
        np.random.seed(42)
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X_normalized[indices].copy()
        
        # Iterate
        for _ in range(self.max_iterations):
            # Assign clusters
            assignments = self._assign_clusters(X_normalized)
            
            # Update centroids
            new_centroids = np.array([
                X_normalized[assignments == k].mean(axis=0) 
                if np.sum(assignments == k) > 0 else self.centroids[k]
                for k in range(self.n_clusters)
            ])
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
        
        return self._assign_clusters(X_normalized).tolist()
    
    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize features to 0-1 range."""
        self._min = X.min(axis=0)
        self._max = X.max(axis=0)
        range_val = self._max - self._min
        range_val[range_val == 0] = 1
        return (X - self._min) / range_val
    
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Assign samples to nearest centroid."""
        distances = np.array([
            np.linalg.norm(X - centroid, axis=1)
            for centroid in self.centroids
        ])
        return distances.argmin(axis=0)
    
    def predict(self, member: AudienceMember) -> int:
        """Predict cluster for a new member."""
        x = np.array([member.features.get(f, 0) for f in self.feature_names])
        x_normalized = (x - self._min) / (self._max - self._min)
        
        distances = [
            np.linalg.norm(x_normalized - centroid)
            for centroid in self.centroids
        ]
        return int(np.argmin(distances))
    
    def get_segment_characteristics(
        self,
        members: List[AudienceMember],
        assignments: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """Get characteristics for each segment."""
        characteristics = {}
        
        for k in range(self.n_clusters):
            segment_members = [m for m, a in zip(members, assignments) if a == k]
            if not segment_members:
                continue
            
            # Calculate feature averages
            avg_features = {}
            for f in self.feature_names:
                values = [m.features.get(f, 0) for m in segment_members]
                avg_features[f] = np.mean(values)
            
            # Identify distinguishing features
            centroid = self.centroids[k]
            overall_avg = self.centroids.mean(axis=0)
            
            distinguishing = []
            for i, f in enumerate(self.feature_names):
                if centroid[i] > overall_avg[i] * 1.2:
                    distinguishing.append(f"High {f}")
                elif centroid[i] < overall_avg[i] * 0.8:
                    distinguishing.append(f"Low {f}")
            
            characteristics[k] = {
                "size": len(segment_members),
                "avg_features": avg_features,
                "distinguishing": distinguishing[:5]
            }
        
        return characteristics


class DBSCANClusterer:
    """
    DBSCAN clustering for density-based segmentation.
    
    Finds clusters of arbitrary shape and identifies outliers.
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels: List[int] = []
    
    def fit(self, members: List[AudienceMember]) -> List[int]:
        """
        Fit DBSCAN on audience members.
        
        Returns:
            Cluster labels (-1 for noise)
        """
        if not members:
            return []
        
        # Extract and normalize features
        feature_names = list(members[0].features.keys())
        X = np.array([
            [m.features.get(f, 0) for f in feature_names]
            for m in members
        ])
        
        X_normalized = (X - X.min(axis=0)) / np.maximum(X.max(axis=0) - X.min(axis=0), 1e-10)
        
        n = len(members)
        labels = [-1] * n
        cluster_id = 0
        visited = [False] * n
        
        for i in range(n):
            if visited[i]:
                continue
            
            visited[i] = True
            neighbors = self._get_neighbors(X_normalized, i)
            
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # Noise
            else:
                self._expand_cluster(X_normalized, labels, i, neighbors, cluster_id, visited)
                cluster_id += 1
        
        self.labels = labels
        return labels
    
    def _get_neighbors(self, X: np.ndarray, idx: int) -> List[int]:
        """Get neighbors within eps distance."""
        distances = np.linalg.norm(X - X[idx], axis=1)
        return np.where(distances <= self.eps)[0].tolist()
    
    def _expand_cluster(
        self,
        X: np.ndarray,
        labels: List[int],
        idx: int,
        neighbors: List[int],
        cluster_id: int,
        visited: List[bool]
    ):
        """Expand cluster from a core point."""
        labels[idx] = cluster_id
        i = 0
        
        while i < len(neighbors):
            neighbor = neighbors[i]
            
            if not visited[neighbor]:
                visited[neighbor] = True
                new_neighbors = self._get_neighbors(X, neighbor)
                
                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend([n for n in new_neighbors if n not in neighbors])
            
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id
            
            i += 1


class LookalikeGenerator:
    """
    Generate lookalike audiences from seed audiences.
    
    Uses similarity scoring to find users similar to high-value customers.
    """
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.seed_centroid: Optional[np.ndarray] = None
        self.seed_variance: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
    
    def fit_seed(self, seed_members: List[AudienceMember]):
        """
        Fit the model on seed audience.
        
        Args:
            seed_members: High-value seed audience
        """
        if not seed_members:
            return
        
        self.feature_names = list(seed_members[0].features.keys())
        X = np.array([
            [m.features.get(f, 0) for f in self.feature_names]
            for m in seed_members
        ])
        
        # Calculate seed centroid and variance
        self.seed_centroid = X.mean(axis=0)
        self.seed_variance = X.var(axis=0) + 1e-10
    
    def score(self, member: AudienceMember) -> float:
        """
        Score a member's similarity to seed audience.
        
        Returns:
            Similarity score (0-1)
        """
        if self.seed_centroid is None:
            return 0.0
        
        x = np.array([member.features.get(f, 0) for f in self.feature_names])
        
        # Mahalanobis-like distance
        diff = x - self.seed_centroid
        distance = np.sqrt(np.sum((diff ** 2) / self.seed_variance))
        
        # Convert to similarity score
        similarity = 1 / (1 + distance)
        return float(similarity)
    
    def generate_lookalike(
        self,
        candidates: List[AudienceMember],
        size: int,
        expansion_factor: float = 1.0
    ) -> List[Tuple[AudienceMember, float]]:
        """
        Generate lookalike audience from candidates.
        
        Args:
            candidates: Pool of potential lookalike members
            size: Target audience size
            expansion_factor: 1.0 = tight match, 5.0 = broad match
            
        Returns:
            List of (member, score) tuples
        """
        # Score all candidates
        scored = [(m, self.score(m)) for m in candidates]
        
        # Adjust threshold based on expansion factor
        adjusted_threshold = self.similarity_threshold / expansion_factor
        
        # Filter and sort
        qualified = [
            (m, s) for m, s in scored
            if s >= adjusted_threshold
        ]
        qualified.sort(key=lambda x: x[1], reverse=True)
        
        return qualified[:size]


class AudienceIntelligenceService:
    """
    Main service for audience intelligence.
    """
    
    def __init__(self):
        self.kmeans = KMeansClusterer()
        self.dbscan = DBSCANClusterer()
        self.lookalike = LookalikeGenerator()
    
    def segment_audience(
        self,
        members: List[AudienceMember],
        method: str = "kmeans",
        n_segments: int = 5
    ) -> Dict[str, Any]:
        """
        Segment audience using specified method.
        
        Args:
            members: Audience members
            method: "kmeans" or "dbscan"
            n_segments: Number of segments (for kmeans)
            
        Returns:
            Segmentation results
        """
        if method == "kmeans":
            self.kmeans.n_clusters = n_segments
            assignments = self.kmeans.fit(members)
            characteristics = self.kmeans.get_segment_characteristics(members, assignments)
        else:
            assignments = self.dbscan.fit(members)
            characteristics = self._get_dbscan_characteristics(members, assignments)
        
        # Build segments
        segments = []
        unique_labels = set(assignments)
        
        for label in unique_labels:
            if label == -1:  # Skip noise for DBSCAN
                continue
            
            segment_members = [m for m, a in zip(members, assignments) if a == label]
            char = characteristics.get(label, {})
            
            segments.append(AudienceSegment(
                segment_id=f"seg_{label}",
                name=self._generate_segment_name(char.get("distinguishing", [])),
                size=len(segment_members),
                centroid=char.get("avg_features", {}),
                characteristics=char.get("distinguishing", []),
                avg_value=np.mean([m.features.get("ltv", 0) for m in segment_members]),
                platforms_synced=[]
            ))
        
        # Update members with segment IDs
        for member, assignment in zip(members, assignments):
            member.segment_id = f"seg_{assignment}"
        
        return {
            "segments": segments,
            "assignments": assignments,
            "total_members": len(members),
            "noise_count": assignments.count(-1) if method == "dbscan" else 0
        }
    
    def _get_dbscan_characteristics(
        self,
        members: List[AudienceMember],
        assignments: List[int]
    ) -> Dict[int, Dict]:
        """Get characteristics for DBSCAN clusters."""
        characteristics = {}
        feature_names = list(members[0].features.keys()) if members else []
        
        for label in set(assignments):
            if label == -1:
                continue
            
            segment_members = [m for m, a in zip(members, assignments) if a == label]
            
            avg_features = {}
            for f in feature_names:
                values = [m.features.get(f, 0) for m in segment_members]
                avg_features[f] = np.mean(values)
            
            characteristics[label] = {
                "size": len(segment_members),
                "avg_features": avg_features,
                "distinguishing": [f"Cluster {label}"]
            }
        
        return characteristics
    
    def _generate_segment_name(self, characteristics: List[str]) -> str:
        """Generate a human-readable segment name."""
        if not characteristics:
            return "General Audience"
        
        # Simple naming based on characteristics
        name_parts = []
        for char in characteristics[:2]:
            if "High" in char:
                name_parts.append(char.replace("High ", "High-"))
            elif "Low" in char:
                name_parts.append(char.replace("Low ", "Low-"))
        
        return " | ".join(name_parts) if name_parts else "Mixed Segment"
    
    def create_lookalike(
        self,
        seed_members: List[AudienceMember],
        candidate_pool: List[AudienceMember],
        target_size: int = 1000,
        expansion: float = 1.0
    ) -> Dict[str, Any]:
        """
        Create lookalike audience from seed.
        
        Args:
            seed_members: High-value seed audience
            candidate_pool: Potential lookalike candidates
            target_size: Target lookalike size
            expansion: Expansion factor (1-5)
            
        Returns:
            Lookalike audience results
        """
        self.lookalike.fit_seed(seed_members)
        lookalikes = self.lookalike.generate_lookalike(
            candidate_pool, target_size, expansion
        )
        
        # Update scores
        for member, score in lookalikes:
            member.lookalike_score = score
        
        return {
            "seed_size": len(seed_members),
            "lookalike_size": len(lookalikes),
            "target_size": target_size,
            "expansion_factor": expansion,
            "avg_similarity": np.mean([s for _, s in lookalikes]) if lookalikes else 0,
            "min_similarity": min([s for _, s in lookalikes]) if lookalikes else 0,
            "members": [m for m, _ in lookalikes]
        }
    
    def get_targeting_params(
        self,
        segment: AudienceSegment,
        platform: str = "meta"
    ) -> Dict[str, Any]:
        """
        Convert segment to platform-specific targeting params.
        
        Args:
            segment: Audience segment
            platform: Target platform (meta, google, tiktok)
            
        Returns:
            Platform-specific targeting parameters
        """
        centroid = segment.centroid
        
        if platform == "meta":
            return {
                "age_min": int(centroid.get("avg_age", 25) - 5),
                "age_max": int(centroid.get("avg_age", 35) + 10),
                "genders": [1, 2],  # All genders
                "geo_locations": {"countries": ["US"]},
                "interests": self._map_to_meta_interests(centroid),
                "behaviors": [],
                "custom_audiences": []
            }
        elif platform == "google":
            return {
                "age_range": "AGE_RANGE_25_34",
                "genders": ["GENDER_UNKNOWN"],
                "parental_status": "PARENTAL_STATUS_UNDETERMINED",
                "affinity_audiences": self._map_to_google_affinities(centroid),
                "in_market_audiences": []
            }
        elif platform == "tiktok":
            return {
                "age_groups": ["AGE_25_34", "AGE_35_44"],
                "genders": ["GENDER_UNLIMITED"],
                "interests": self._map_to_tiktok_interests(centroid),
                "behaviors": []
            }
        
        return {}
    
    def _map_to_meta_interests(self, centroid: Dict[str, float]) -> List[Dict]:
        """Map centroid features to Meta interests."""
        interests = []
        
        if centroid.get("shopping_frequency", 0) > 0.5:
            interests.append({"id": "6003139266461", "name": "Shopping (Hobby)"})
        if centroid.get("tech_affinity", 0) > 0.5:
            interests.append({"id": "6003384745172", "name": "Technology"})
        if centroid.get("fitness_score", 0) > 0.5:
            interests.append({"id": "6003107902433", "name": "Fitness and wellness"})
        
        return interests
    
    def _map_to_google_affinities(self, centroid: Dict[str, float]) -> List[str]:
        """Map centroid features to Google affinity audiences."""
        affinities = []
        
        if centroid.get("shopping_frequency", 0) > 0.5:
            affinities.append("Shoppers/Value Shoppers")
        if centroid.get("tech_affinity", 0) > 0.5:
            affinities.append("Technology/Technophiles")
        
        return affinities
    
    def _map_to_tiktok_interests(self, centroid: Dict[str, float]) -> List[Dict]:
        """Map centroid features to TikTok interests."""
        return [{"interest_id": "123", "interest_name": "Fashion"}]


# Singleton
_audience_service: Optional[AudienceIntelligenceService] = None


def get_audience_service() -> AudienceIntelligenceService:
    """Get or create audience intelligence service."""
    global _audience_service
    if _audience_service is None:
        _audience_service = AudienceIntelligenceService()
    return _audience_service
