"""
Multi-Touch Attribution ML Models

Shapley values, Markov chains, and path analysis for attribution.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import numpy as np
from itertools import combinations


@dataclass
class Touchpoint:
    """A single touchpoint in a customer journey."""
    touchpoint_id: str
    customer_id: str
    channel: str
    campaign_id: Optional[str]
    timestamp: datetime
    interaction_type: str  # impression, click, conversion
    value: float = 0.0


@dataclass
class CustomerJourney:
    """Complete customer journey to conversion."""
    customer_id: str
    touchpoints: List[Touchpoint]
    conversion_value: float
    converted: bool


@dataclass
class ChannelAttribution:
    """Attribution results for a channel."""
    channel: str
    shapley_value: float
    markov_value: float
    first_touch: float
    last_touch: float
    linear: float
    time_decay: float


class ShapleyCalculator:
    """
    Compute Shapley value-based attribution.
    
    Shapley values fairly distribute credit based on each channel's
    marginal contribution to conversions.
    """
    
    def __init__(self, conversion_rate_func=None):
        self.conversion_rate_func = conversion_rate_func or self._default_conversion_rate
        self._coalition_cache: Dict[frozenset, float] = {}
    
    def _default_conversion_rate(self, channels: Set[str], journeys: List[CustomerJourney]) -> float:
        """Calculate conversion rate for a coalition of channels."""
        if not channels:
            return 0.0
        
        # Count conversions where journey contains only these channels
        converted = 0
        total = 0
        
        for journey in journeys:
            journey_channels = set(t.channel for t in journey.touchpoints)
            if journey_channels.issubset(channels) or journey_channels.intersection(channels):
                total += 1
                if journey.converted:
                    converted += 1
        
        return converted / max(total, 1)
    
    def compute(
        self,
        journeys: List[CustomerJourney]
    ) -> Dict[str, float]:
        """
        Compute Shapley values for all channels.
        
        Args:
            journeys: List of customer journeys
            
        Returns:
            Dictionary mapping channel names to Shapley values
        """
        # Get all unique channels
        all_channels = set()
        for journey in journeys:
            for tp in journey.touchpoints:
                all_channels.add(tp.channel)
        
        n = len(all_channels)
        channels_list = list(all_channels)
        shapley_values = {c: 0.0 for c in channels_list}
        
        # For each channel, compute its marginal contribution
        for channel in channels_list:
            other_channels = all_channels - {channel}
            marginal_sum = 0.0
            
            # Iterate over all possible coalitions without this channel
            for size in range(len(other_channels) + 1):
                for coalition in combinations(other_channels, size):
                    coalition_set = set(coalition)
                    coalition_with = coalition_set | {channel}
                    
                    # Get conversion rates
                    v_with = self._get_coalition_value(frozenset(coalition_with), journeys)
                    v_without = self._get_coalition_value(frozenset(coalition_set), journeys)
                    
                    # Calculate weight
                    weight = (np.math.factorial(size) * np.math.factorial(n - size - 1)) / np.math.factorial(n)
                    
                    # Add weighted marginal contribution
                    marginal_sum += weight * (v_with - v_without)
            
            shapley_values[channel] = marginal_sum
        
        # Normalize to sum to 1
        total = sum(shapley_values.values())
        if total > 0:
            shapley_values = {k: v / total for k, v in shapley_values.items()}
        
        return shapley_values
    
    def _get_coalition_value(self, coalition: frozenset, journeys: List[CustomerJourney]) -> float:
        """Get cached or compute coalition value."""
        if coalition not in self._coalition_cache:
            self._coalition_cache[coalition] = self._default_conversion_rate(set(coalition), journeys)
        return self._coalition_cache[coalition]


class MarkovChainAttributor:
    """
    Markov Chain-based attribution model.
    
    Models customer journeys as a Markov chain and calculates
    channel importance based on removal effects.
    """
    
    def __init__(self):
        self.transition_matrix: Dict[str, Dict[str, float]] = {}
        self.channels: Set[str] = set()
    
    def fit(self, journeys: List[CustomerJourney]):
        """
        Fit the Markov Chain model on customer journeys.
        
        Args:
            journeys: List of customer journeys
        """
        # Count transitions
        transition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for journey in journeys:
            touchpoints = journey.touchpoints
            if not touchpoints:
                continue
            
            # Add start state
            prev_channel = "START"
            
            for tp in touchpoints:
                channel = tp.channel
                self.channels.add(channel)
                transition_counts[prev_channel][channel] += 1
                prev_channel = channel
            
            # Add end state
            end_state = "CONVERSION" if journey.converted else "NULL"
            transition_counts[prev_channel][end_state] += 1
        
        # Convert counts to probabilities
        for from_state, to_states in transition_counts.items():
            total = sum(to_states.values())
            self.transition_matrix[from_state] = {
                to_state: count / total
                for to_state, count in to_states.items()
            }
    
    def compute_attribution(self) -> Dict[str, float]:
        """
        Compute attribution using removal effect method.
        
        Returns:
            Dictionary mapping channels to attribution values
        """
        # Calculate base conversion probability
        base_conv_rate = self._simulate_conversions()
        
        # Calculate removal effect for each channel
        removal_effects = {}
        for channel in self.channels:
            removed_rate = self._simulate_conversions_without(channel)
            removal_effects[channel] = base_conv_rate - removed_rate
        
        # Normalize
        total = sum(max(0, v) for v in removal_effects.values())
        if total > 0:
            return {k: max(0, v) / total for k, v in removal_effects.items()}
        
        # Equal attribution if no removal effects
        n = len(self.channels)
        return {c: 1 / n for c in self.channels}
    
    def _simulate_conversions(self, excluded_channel: Optional[str] = None) -> float:
        """Simulate conversion probability through the chain."""
        if not self.transition_matrix:
            return 0.0
        
        # Simple simulation: probability of reaching CONVERSION from START
        current = "START"
        prob = 1.0
        visited = set()
        max_steps = 20
        
        for _ in range(max_steps):
            if current in visited or current in ["CONVERSION", "NULL"]:
                break
            visited.add(current)
            
            transitions = self.transition_matrix.get(current, {})
            if not transitions:
                break
            
            # Filter out excluded channel
            if excluded_channel:
                transitions = {k: v for k, v in transitions.items() if k != excluded_channel}
                if not transitions:
                    return 0.0
                # Renormalize
                total = sum(transitions.values())
                transitions = {k: v / total for k, v in transitions.items()}
            
            # Get conversion probability
            if "CONVERSION" in transitions:
                return prob * transitions["CONVERSION"]
            
            # Weighted average of continuing
            total_continue = sum(v for k, v in transitions.items() if k not in ["CONVERSION", "NULL"])
            if total_continue == 0:
                break
            
            # Move to most likely next state (simplified)
            next_state = max(
                ((k, v) for k, v in transitions.items() if k not in ["CONVERSION", "NULL"]),
                key=lambda x: x[1],
                default=(None, 0)
            )
            if next_state[0] is None:
                break
            
            prob *= next_state[1]
            current = next_state[0]
        
        return 0.0
    
    def _simulate_conversions_without(self, excluded_channel: str) -> float:
        """Simulate conversions with a channel removed."""
        return self._simulate_conversions(excluded_channel)


class PathAnalyzer:
    """
    Analyze customer journey paths for patterns.
    """
    
    def __init__(self):
        self.path_counts: Dict[str, int] = defaultdict(int)
        self.path_values: Dict[str, float] = defaultdict(float)
    
    def analyze(self, journeys: List[CustomerJourney]) -> Dict[str, Any]:
        """
        Analyze journey paths.
        
        Args:
            journeys: Customer journeys
            
        Returns:
            Path analysis results
        """
        for journey in journeys:
            if not journey.converted:
                continue
            
            # Create path string
            path = " → ".join(t.channel for t in journey.touchpoints)
            self.path_counts[path] += 1
            self.path_values[path] += journey.conversion_value
        
        # Get top paths
        sorted_paths = sorted(
            self.path_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        top_paths = [
            {
                "path": path,
                "conversions": count,
                "value": self.path_values[path],
                "avg_value": self.path_values[path] / count
            }
            for path, count in sorted_paths
        ]
        
        return {
            "total_paths": len(self.path_counts),
            "total_conversions": sum(self.path_counts.values()),
            "total_value": sum(self.path_values.values()),
            "top_paths": top_paths,
            "path_length_distribution": self._get_path_length_dist()
        }
    
    def _get_path_length_dist(self) -> Dict[int, int]:
        """Get distribution of path lengths."""
        dist = defaultdict(int)
        for path, count in self.path_counts.items():
            length = path.count(" → ") + 1
            dist[length] += count
        return dict(sorted(dist.items()))


class AttributionEngine:
    """
    Main attribution service combining all models.
    """
    
    def __init__(self):
        self.shapley = ShapleyCalculator()
        self.markov = MarkovChainAttributor()
        self.path_analyzer = PathAnalyzer()
    
    def compute_attribution(
        self,
        journeys: List[CustomerJourney]
    ) -> Dict[str, ChannelAttribution]:
        """
        Compute full attribution across all models.
        
        Args:
            journeys: Customer journeys
            
        Returns:
            Attribution results per channel
        """
        # Fit Markov model
        self.markov.fit(journeys)
        
        # Compute different attribution models
        shapley_values = self.shapley.compute(journeys)
        markov_values = self.markov.compute_attribution()
        first_touch = self._first_touch_attribution(journeys)
        last_touch = self._last_touch_attribution(journeys)
        linear = self._linear_attribution(journeys)
        time_decay = self._time_decay_attribution(journeys)
        
        # Combine results
        all_channels = set(shapley_values.keys()) | set(markov_values.keys())
        
        results = {}
        for channel in all_channels:
            results[channel] = ChannelAttribution(
                channel=channel,
                shapley_value=shapley_values.get(channel, 0),
                markov_value=markov_values.get(channel, 0),
                first_touch=first_touch.get(channel, 0),
                last_touch=last_touch.get(channel, 0),
                linear=linear.get(channel, 0),
                time_decay=time_decay.get(channel, 0)
            )
        
        return results
    
    def _first_touch_attribution(self, journeys: List[CustomerJourney]) -> Dict[str, float]:
        """First touch attribution."""
        counts = defaultdict(float)
        total = 0.0
        
        for journey in journeys:
            if journey.converted and journey.touchpoints:
                first_channel = journey.touchpoints[0].channel
                counts[first_channel] += journey.conversion_value
                total += journey.conversion_value
        
        return {k: v / max(total, 1) for k, v in counts.items()}
    
    def _last_touch_attribution(self, journeys: List[CustomerJourney]) -> Dict[str, float]:
        """Last touch attribution."""
        counts = defaultdict(float)
        total = 0.0
        
        for journey in journeys:
            if journey.converted and journey.touchpoints:
                last_channel = journey.touchpoints[-1].channel
                counts[last_channel] += journey.conversion_value
                total += journey.conversion_value
        
        return {k: v / max(total, 1) for k, v in counts.items()}
    
    def _linear_attribution(self, journeys: List[CustomerJourney]) -> Dict[str, float]:
        """Linear attribution - equal credit to all touchpoints."""
        counts = defaultdict(float)
        total = 0.0
        
        for journey in journeys:
            if journey.converted and journey.touchpoints:
                n = len(journey.touchpoints)
                credit_per_touch = journey.conversion_value / n
                
                for tp in journey.touchpoints:
                    counts[tp.channel] += credit_per_touch
                    total += credit_per_touch
        
        return {k: v / max(total, 1) for k, v in counts.items()}
    
    def _time_decay_attribution(
        self,
        journeys: List[CustomerJourney],
        half_life_days: float = 7.0
    ) -> Dict[str, float]:
        """Time decay attribution - more credit to recent touchpoints."""
        counts = defaultdict(float)
        total = 0.0
        
        for journey in journeys:
            if not journey.converted or not journey.touchpoints:
                continue
            
            conversion_time = journey.touchpoints[-1].timestamp
            weights = []
            
            for tp in journey.touchpoints:
                days_before = (conversion_time - tp.timestamp).total_seconds() / 86400
                weight = 2 ** (-days_before / half_life_days)
                weights.append((tp.channel, weight))
            
            total_weight = sum(w for _, w in weights)
            for channel, weight in weights:
                credit = (weight / total_weight) * journey.conversion_value
                counts[channel] += credit
                total += credit
        
        return {k: v / max(total, 1) for k, v in counts.items()}


# Singleton
_attribution_engine: Optional[AttributionEngine] = None


def get_attribution_engine() -> AttributionEngine:
    """Get or create attribution engine instance."""
    global _attribution_engine
    if _attribution_engine is None:
        _attribution_engine = AttributionEngine()
    return _attribution_engine
