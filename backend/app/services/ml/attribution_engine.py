"""
Attribution Engine ML Model

Uses Shapley values and Markov chains for multi-touch attribution.
Integrated with MLflow for model versioning and tracking.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, date
from collections import defaultdict
import itertools
import math

try:
    import numpy as np
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    np = None

from app.services.mlops import MLflowClient, ModelType


@dataclass
class Touchpoint:
    """A single marketing touchpoint in a customer journey."""
    timestamp: datetime
    channel: str
    campaign_id: Optional[str] = None
    cost: float = 0.0
    
    def __hash__(self):
        return hash((self.timestamp, self.channel))


@dataclass
class ConversionPath:
    """A complete customer conversion path."""
    path_id: str
    touchpoints: List[Touchpoint]
    converted: bool
    conversion_value: float = 0.0
    
    @property
    def channels(self) -> List[str]:
        return [t.channel for t in self.touchpoints]
    
    @property
    def total_cost(self) -> float:
        return sum(t.cost for t in self.touchpoints)


@dataclass
class ChannelAttribution:
    """Attribution results for a channel."""
    channel: str
    shapley_value: float
    markov_value: float
    first_touch: float
    last_touch: float
    linear: float
    contribution_percentage: float
    total_conversions: int
    total_value: float


class ShapleyCalculator:
    """
    Calculate Shapley values for fair attribution.
    
    Shapley values are a game-theoretic approach that considers
    the marginal contribution of each channel across all possible orderings.
    """
    
    def __init__(self, max_channels: int = 10):
        self.max_channels = max_channels
        self._factorial_cache = {}
    
    def _factorial(self, n: int) -> int:
        if n in self._factorial_cache:
            return self._factorial_cache[n]
        result = 1 if n <= 1 else n * self._factorial(n - 1)
        self._factorial_cache[n] = result
        return result
    
    def calculate(self, paths: List[ConversionPath]) -> Dict[str, float]:
        """
        Calculate Shapley values for all channels.
        
        Uses the coalition game theory approach where:
        - Value of empty coalition = 0
        - Value of grand coalition = total conversions
        - Each channel gets credit based on marginal contribution
        """
        if not paths:
            return {}
        
        # Get all unique channels
        all_channels = set()
        for path in paths:
            all_channels.update(path.channels)
        
        all_channels = list(all_channels)[:self.max_channels]
        n = len(all_channels)
        
        if n == 0:
            return {}
        
        # Calculate value function for each coalition
        coalition_values = self._calculate_coalition_values(paths, all_channels)
        
        # Calculate Shapley values
        shapley = {}
        for channel in all_channels:
            shapley[channel] = self._shapley_value(channel, all_channels, coalition_values)
        
        # Normalize to sum to 1
        total = sum(shapley.values())
        if total > 0:
            shapley = {k: v / total for k, v in shapley.items()}
        
        return shapley
    
    def _calculate_coalition_values(
        self, 
        paths: List[ConversionPath],
        channels: List[str]
    ) -> Dict[frozenset, float]:
        """Calculate conversion value for each possible coalition of channels."""
        values = {}
        
        # Generate all possible coalitions
        for r in range(len(channels) + 1):
            for coalition in itertools.combinations(channels, r):
                coalition_set = frozenset(coalition)
                values[coalition_set] = self._coalition_value(paths, coalition_set)
        
        return values
    
    def _coalition_value(
        self, 
        paths: List[ConversionPath], 
        coalition: frozenset
    ) -> float:
        """Calculate value if only channels in coalition were present."""
        if not coalition:
            return 0.0
        
        value = 0.0
        for path in paths:
            if path.converted:
                # Check if any touchpoint in path is in the coalition
                path_channels = set(path.channels)
                if path_channels & coalition:
                    # Proportion of path covered by coalition
                    coverage = len(path_channels & coalition) / len(path_channels)
                    value += path.conversion_value * coverage
        
        return value
    
    def _shapley_value(
        self,
        channel: str,
        all_channels: List[str],
        coalition_values: Dict[frozenset, float]
    ) -> float:
        """Calculate Shapley value for a single channel."""
        n = len(all_channels)
        other_channels = [c for c in all_channels if c != channel]
        
        shapley = 0.0
        
        # Iterate over all subsets not containing the channel
        for r in range(len(other_channels) + 1):
            for subset in itertools.combinations(other_channels, r):
                subset_set = frozenset(subset)
                with_channel = frozenset(subset + (channel,))
                
                # Marginal contribution
                marginal = coalition_values.get(with_channel, 0) - coalition_values.get(subset_set, 0)
                
                # Weight by permutation count
                weight = (self._factorial(r) * self._factorial(n - r - 1)) / self._factorial(n)
                shapley += weight * marginal
        
        return shapley


class MarkovChainAttribution:
    """
    Markov chain based attribution model.
    
    Models customer journeys as a Markov chain where:
    - States are channels
    - Transitions are movements between touchpoints
    - Removes each channel to calculate removal effect
    """
    
    def __init__(self):
        self.transition_matrix: Dict[str, Dict[str, float]] = {}
        self.channels: Set[str] = set()
    
    def fit(self, paths: List[ConversionPath]):
        """Fit the Markov chain on conversion paths."""
        # Initialize counts
        transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.channels = set()
        
        for path in paths:
            channels = ["START"] + path.channels + (["CONVERSION"] if path.converted else ["NULL"])
            self.channels.update(path.channels)
            
            for i in range(len(channels) - 1):
                transitions[channels[i]][channels[i + 1]] += 1
        
        # Convert to probabilities
        self.transition_matrix = {}
        for from_state, to_states in transitions.items():
            total = sum(to_states.values())
            self.transition_matrix[from_state] = {
                to: count / total for to, count in to_states.items()
            }
    
    def calculate_removal_effects(self) -> Dict[str, float]:
        """Calculate attribution by removal effect."""
        if not self.transition_matrix:
            return {}
        
        # Base conversion rate
        base_rate = self._calculate_conversion_rate(self.transition_matrix)
        
        removal_effects = {}
        for channel in self.channels:
            # Create matrix without this channel
            reduced_matrix = self._remove_channel(channel)
            reduced_rate = self._calculate_conversion_rate(reduced_matrix)
            
            # Removal effect
            removal_effects[channel] = max(0, base_rate - reduced_rate)
        
        # Normalize
        total = sum(removal_effects.values())
        if total > 0:
            removal_effects = {k: v / total for k, v in removal_effects.items()}
        
        return removal_effects
    
    def _remove_channel(self, channel: str) -> Dict[str, Dict[str, float]]:
        """Create transition matrix with a channel removed."""
        new_matrix = {}
        
        for from_state, transitions in self.transition_matrix.items():
            if from_state == channel:
                continue
            
            new_transitions = {}
            removed_prob = transitions.get(channel, 0)
            remaining_prob = 1 - removed_prob
            
            for to_state, prob in transitions.items():
                if to_state != channel:
                    # Redistribute probability
                    new_prob = prob / remaining_prob if remaining_prob > 0 else 0
                    new_transitions[to_state] = new_prob
            
            if new_transitions:
                new_matrix[from_state] = new_transitions
        
        return new_matrix
    
    def _calculate_conversion_rate(
        self, 
        matrix: Dict[str, Dict[str, float]],
        max_steps: int = 100
    ) -> float:
        """Calculate conversion probability from START to CONVERSION."""
        if "START" not in matrix:
            return 0.0
        
        # Simple simulation approach
        conversion_prob = 0.0
        current_probs = {"START": 1.0}
        
        for _ in range(max_steps):
            next_probs = defaultdict(float)
            
            for state, prob in current_probs.items():
                if state == "CONVERSION":
                    conversion_prob += prob
                elif state == "NULL":
                    continue
                elif state in matrix:
                    for next_state, trans_prob in matrix[state].items():
                        next_probs[next_state] += prob * trans_prob
            
            current_probs = dict(next_probs)
            
            # Check convergence
            if sum(current_probs.values()) < 0.001:
                break
        
        return conversion_prob


class AttributionEngine:
    """
    Main attribution engine combining multiple models.
    
    Provides:
    - Shapley value attribution
    - Markov chain attribution
    - First/last touch attribution
    - Linear attribution
    - Budget recommendations
    """
    
    def __init__(self, mlflow_client: Optional[MLflowClient] = None):
        self.mlflow_client = mlflow_client or MLflowClient()
        self.shapley_calc = ShapleyCalculator()
        self.markov = MarkovChainAttribution()
    
    def calculate_attribution(
        self,
        paths: List[ConversionPath],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Calculate multi-model attribution for all channels.
        
        Returns comprehensive attribution report with multiple models.
        """
        if not paths:
            return {"channels": [], "total_conversions": 0, "total_value": 0}
        
        converted_paths = [p for p in paths if p.converted]
        
        # Calculate all attribution models
        shapley = self.shapley_calc.calculate(converted_paths)
        
        self.markov.fit(paths)
        markov = self.markov.calculate_removal_effects()
        
        first_touch = self._first_touch_attribution(converted_paths)
        last_touch = self._last_touch_attribution(converted_paths)
        linear = self._linear_attribution(converted_paths)
        
        # Aggregate channel stats
        channel_stats = self._aggregate_channel_stats(paths)
        
        # Combine into channel reports
        all_channels = set(shapley.keys()) | set(markov.keys())
        total_value = sum(p.conversion_value for p in converted_paths)
        
        attributions = []
        for channel in all_channels:
            attr = ChannelAttribution(
                channel=channel,
                shapley_value=shapley.get(channel, 0),
                markov_value=markov.get(channel, 0),
                first_touch=first_touch.get(channel, 0),
                last_touch=last_touch.get(channel, 0),
                linear=linear.get(channel, 0),
                contribution_percentage=shapley.get(channel, 0) * 100,
                total_conversions=channel_stats.get(channel, {}).get("conversions", 0),
                total_value=total_value * shapley.get(channel, 0)
            )
            attributions.append(attr)
        
        # Sort by Shapley value
        attributions.sort(key=lambda x: x.shapley_value, reverse=True)
        
        # Log to MLflow
        self._log_attribution(attributions)
        
        return {
            "channels": [self._attribution_to_dict(a) for a in attributions],
            "total_conversions": len(converted_paths),
            "total_value": total_value,
            "avg_touchpoints": sum(len(p.touchpoints) for p in paths) / len(paths),
            "insights": self._generate_insights(attributions)
        }
    
    def _first_touch_attribution(self, paths: List[ConversionPath]) -> Dict[str, float]:
        """Calculate first-touch attribution."""
        counts = defaultdict(float)
        total = 0
        
        for path in paths:
            if path.touchpoints:
                counts[path.touchpoints[0].channel] += path.conversion_value
                total += path.conversion_value
        
        if total == 0:
            return {}
        return {k: v / total for k, v in counts.items()}
    
    def _last_touch_attribution(self, paths: List[ConversionPath]) -> Dict[str, float]:
        """Calculate last-touch attribution."""
        counts = defaultdict(float)
        total = 0
        
        for path in paths:
            if path.touchpoints:
                counts[path.touchpoints[-1].channel] += path.conversion_value
                total += path.conversion_value
        
        if total == 0:
            return {}
        return {k: v / total for k, v in counts.items()}
    
    def _linear_attribution(self, paths: List[ConversionPath]) -> Dict[str, float]:
        """Calculate linear attribution (equal credit)."""
        counts = defaultdict(float)
        total = 0
        
        for path in paths:
            n = len(path.touchpoints)
            if n > 0:
                credit = path.conversion_value / n
                for tp in path.touchpoints:
                    counts[tp.channel] += credit
                total += path.conversion_value
        
        if total == 0:
            return {}
        return {k: v / total for k, v in counts.items()}
    
    def _aggregate_channel_stats(
        self, 
        paths: List[ConversionPath]
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate statistics per channel."""
        stats = defaultdict(lambda: {"touchpoints": 0, "conversions": 0, "cost": 0})
        
        for path in paths:
            for tp in path.touchpoints:
                stats[tp.channel]["touchpoints"] += 1
                stats[tp.channel]["cost"] += tp.cost
            
            if path.converted:
                for channel in set(path.channels):
                    stats[channel]["conversions"] += 1
        
        return dict(stats)
    
    def _attribution_to_dict(self, attr: ChannelAttribution) -> Dict[str, Any]:
        """Convert attribution to API format."""
        return {
            "channel": attr.channel,
            "shapley_attribution": round(attr.shapley_value, 4),
            "markov_attribution": round(attr.markov_value, 4),
            "first_touch_attribution": round(attr.first_touch, 4),
            "last_touch_attribution": round(attr.last_touch, 4),
            "contribution_percentage": round(attr.contribution_percentage, 2),
            "total_conversions": attr.total_conversions,
            "total_value": round(attr.total_value, 2)
        }
    
    def _generate_insights(self, attributions: List[ChannelAttribution]) -> List[str]:
        """Generate insights from attribution results."""
        insights = []
        
        if not attributions:
            return insights
        
        # Top performer
        top = attributions[0]
        insights.append(
            f"{top.channel} is the top contributor with {top.contribution_percentage:.1f}% attribution"
        )
        
        # First touch vs last touch discrepancy
        for attr in attributions[:3]:
            if abs(attr.first_touch - attr.last_touch) > 0.15:
                if attr.first_touch > attr.last_touch:
                    insights.append(
                        f"{attr.channel} is stronger at awareness (first touch) than conversion (last touch)"
                    )
                else:
                    insights.append(
                        f"{attr.channel} is stronger at conversion (last touch) than awareness (first touch)"
                    )
        
        # Model disagreement
        for attr in attributions:
            if abs(attr.shapley_value - attr.markov_value) > 0.1:
                insights.append(
                    f"Models disagree on {attr.channel} - consider deeper analysis"
                )
        
        return insights[:5]
    
    def get_budget_recommendations(
        self,
        attributions: List[ChannelAttribution],
        current_budgets: Dict[str, float],
        total_budget: float
    ) -> Dict[str, float]:
        """Generate budget allocation recommendations based on attribution."""
        # Weight by Shapley value
        total_shapley = sum(a.shapley_value for a in attributions)
        
        if total_shapley == 0:
            return current_budgets
        
        recommendations = {}
        for attr in attributions:
            optimal_share = attr.shapley_value / total_shapley
            recommendations[attr.channel] = total_budget * optimal_share
        
        return recommendations
    
    def _log_attribution(self, attributions: List[ChannelAttribution]):
        """Log attribution results to MLflow."""
        try:
            experiment = self.mlflow_client.get_experiment("attribution")
            if not experiment:
                experiment = self.mlflow_client.create_experiment("attribution")
            
            run = self.mlflow_client.start_run(
                experiment.experiment_id,
                f"attribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            for attr in attributions[:5]:
                self.mlflow_client.log_metric(
                    run.run_id,
                    f"{attr.channel}_shapley",
                    attr.shapley_value
                )
            
            self.mlflow_client.end_run(run.run_id)
        except Exception:
            pass


# Singleton instance
_engine: Optional[AttributionEngine] = None


def get_attribution_engine() -> AttributionEngine:
    """Get or create the attribution engine instance."""
    global _engine
    if _engine is None:
        _engine = AttributionEngine()
    return _engine
