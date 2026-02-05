"""
ROAS Optimizer ML Models

Real-time budget optimization using Thompson Sampling and LSTM forecasting.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CampaignState:
    """Current state of a campaign for optimization."""
    campaign_id: str
    current_budget: float
    spend_today: float
    impressions: int
    clicks: int
    conversions: int
    revenue: float
    ctr: float
    cvr: float
    roas: float
    hour_of_day: int
    day_of_week: int


@dataclass
class BudgetDecision:
    """Budget decision from optimizer."""
    campaign_id: str
    current_budget: float
    recommended_budget: float
    action: str  # increase, decrease, pause, maintain
    predicted_roas: float
    confidence: float
    reasoning: str


class ThompsonSamplingOptimizer:
    """
    Multi-Armed Bandit optimizer using Thompson Sampling.
    
    Models each campaign's success probability as a Beta distribution.
    Samples from posteriors to select budget allocation.
    """
    
    def __init__(
        self,
        success_threshold: float = 1.0,  # ROAS > 1.0 = success
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ):
        self.success_threshold = success_threshold
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        # Beta distribution parameters per campaign
        self.alphas: Dict[str, float] = {}
        self.betas: Dict[str, float] = {}
        self.observation_counts: Dict[str, int] = {}
    
    def initialize_campaign(self, campaign_id: str):
        """Initialize a new campaign with prior."""
        if campaign_id not in self.alphas:
            self.alphas[campaign_id] = self.prior_alpha
            self.betas[campaign_id] = self.prior_beta
            self.observation_counts[campaign_id] = 0
    
    def update(self, campaign_id: str, roas: float):
        """
        Update posterior based on observed ROAS.
        
        Args:
            campaign_id: Campaign identifier
            roas: Observed ROAS value
        """
        self.initialize_campaign(campaign_id)
        
        # Convert ROAS to success/failure
        if roas > self.success_threshold:
            self.alphas[campaign_id] += 1
        else:
            self.betas[campaign_id] += 1
        
        self.observation_counts[campaign_id] += 1
    
    def sample(self, campaign_id: str) -> float:
        """Sample success probability from posterior."""
        self.initialize_campaign(campaign_id)
        return np.random.beta(
            self.alphas[campaign_id],
            self.betas[campaign_id]
        )
    
    def get_allocation(
        self,
        campaign_ids: List[str],
        total_budget: float,
        min_budget: float = 10.0
    ) -> Dict[str, float]:
        """
        Get budget allocation across campaigns.
        
        Uses Thompson Sampling to probabilistically allocate budget
        based on expected performance.
        """
        n_samples = 1000
        allocations = {cid: 0 for cid in campaign_ids}
        
        for _ in range(n_samples):
            # Sample from each campaign's posterior
            samples = {
                cid: self.sample(cid) 
                for cid in campaign_ids
            }
            
            # Winner takes all for this sample
            winner = max(samples, key=samples.get)
            allocations[winner] += 1
        
        # Normalize to budget
        total_wins = sum(allocations.values())
        result = {}
        remaining_budget = total_budget
        
        for cid in campaign_ids[:-1]:
            budget = max(
                min_budget,
                (allocations[cid] / total_wins) * total_budget
            )
            budget = min(budget, remaining_budget - min_budget * (len(campaign_ids) - len(result) - 1))
            result[cid] = budget
            remaining_budget -= budget
        
        # Last campaign gets remainder
        result[campaign_ids[-1]] = remaining_budget
        
        return result
    
    def get_exploration_score(self, campaign_id: str) -> float:
        """Get exploration priority score (higher = less explored)."""
        self.initialize_campaign(campaign_id)
        n = self.observation_counts[campaign_id]
        if n == 0:
            return 1.0
        return 1.0 / np.sqrt(n)


class ROASForecaster(nn.Module):
    """
    LSTM-based ROAS forecasting model.
    
    Predicts next-hour/24-hour ROAS based on historical metrics.
    """
    
    def __init__(
        self,
        input_features: int = 15,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Feature names for reference
        self.feature_names = [
            'impressions', 'clicks', 'conversions', 'spend', 'revenue',
            'ctr', 'cvr', 'cpc', 'cpm', 'roas',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'budget_util'
        ]
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Output layers
        self.output = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 1h, 6h, 24h predictions
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, features)
            return_attention: If True, also return attention weights
            
        Returns:
            Predictions (batch, 3) for 1h, 6h, 24h ROAS
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        
        # Attention weights
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Predict
        predictions = self.output(context)
        
        if return_attention:
            return predictions, attn_weights.squeeze(-1)
        return predictions


class PPOBudgetController(nn.Module):
    """
    PPO (Proximal Policy Optimization) agent for budget control.
    
    State: Campaign metrics + context
    Action: Budget adjustment factor
    Reward: ROAS improvement
    """
    
    def __init__(
        self,
        state_dim: int = 20,
        action_dim: int = 5  # Discrete: -50%, -25%, 0%, +25%, +50%
    ):
        super().__init__()
        
        self.action_space = [-0.5, -0.25, 0.0, 0.25, 0.5]
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            action_probs: Action probabilities
            value: State value estimate
        """
        features = self.features(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
    
    def select_action(self, state: torch.Tensor) -> Tuple[int, float]:
        """
        Select action for a state.
        
        Returns:
            action_idx: Selected action index
            adjustment: Budget adjustment factor
        """
        with torch.no_grad():
            probs, _ = self.forward(state)
            action_idx = torch.multinomial(probs, 1).item()
            adjustment = self.action_space[action_idx]
        return action_idx, adjustment


class ROASOptimizerService:
    """
    High-level service for real-time ROAS optimization.
    
    Combines Thompson Sampling, LSTM forecasting, and RL for
    comprehensive budget optimization.
    """
    
    def __init__(
        self,
        min_budget: float = 10.0,
        max_budget_increase: float = 2.0,
        max_budget_decrease: float = 0.5,
        pause_threshold: float = 0.5
    ):
        self.min_budget = min_budget
        self.max_increase = max_budget_increase
        self.max_decrease = max_budget_decrease
        self.pause_threshold = pause_threshold
        
        # Initialize models
        self.bandit = ThompsonSamplingOptimizer()
        self.forecaster = ROASForecaster()
        self.ppo = PPOBudgetController()
        
        # Set to eval mode
        self.forecaster.eval()
        self.ppo.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.forecaster.to(self.device)
        self.ppo.to(self.device)
    
    def optimize(
        self,
        campaigns: List[CampaignState],
        historical_metrics: Dict[str, np.ndarray],
        dry_run: bool = True
    ) -> List[BudgetDecision]:
        """
        Run optimization cycle for all campaigns.
        
        Args:
            campaigns: Current campaign states
            historical_metrics: 24h of 30-min metrics per campaign
            dry_run: If True, don't apply changes
            
        Returns:
            List of budget decisions
        """
        decisions = []
        
        for campaign in campaigns:
            # Update bandit with current ROAS
            self.bandit.update(campaign.campaign_id, campaign.roas)
            
            # Forecast future ROAS
            metrics = historical_metrics.get(campaign.campaign_id)
            if metrics is not None:
                forecast = self._forecast_roas(metrics)
            else:
                forecast = {'1h': campaign.roas, '6h': campaign.roas, '24h': campaign.roas}
            
            # Make decision
            decision = self._make_decision(campaign, forecast)
            decisions.append(decision)
        
        return decisions
    
    def _forecast_roas(self, metrics: np.ndarray) -> Dict[str, float]:
        """Forecast ROAS using LSTM model."""
        with torch.no_grad():
            x = torch.FloatTensor(metrics).unsqueeze(0).to(self.device)
            predictions = self.forecaster(x)
            return {
                '1h': predictions[0, 0].item(),
                '6h': predictions[0, 1].item(),
                '24h': predictions[0, 2].item()
            }
    
    def _make_decision(
        self,
        campaign: CampaignState,
        forecast: Dict[str, float]
    ) -> BudgetDecision:
        """Make budget decision for a campaign."""
        current = campaign.current_budget
        predicted_roas = forecast['1h']
        
        # Decision logic
        if predicted_roas < self.pause_threshold:
            # Pause underperformers
            action = 'pause'
            new_budget = 0.0
            reasoning = f"Predicted ROAS ({predicted_roas:.2f}) below pause threshold ({self.pause_threshold})"
            confidence = 0.9
            
        elif predicted_roas > 3.0 and campaign.roas > 2.0:
            # Scale winners aggressively
            increase = min(0.5, self.max_increase - 1.0)
            new_budget = min(current * (1 + increase), current * self.max_increase)
            action = 'increase'
            reasoning = f"High ROAS ({campaign.roas:.2f}) with stable forecast. Scaling by {increase*100:.0f}%"
            confidence = 0.85
            
        elif predicted_roas < 1.0:
            # Reduce poor performers
            decrease = min(0.25, 1.0 - self.max_decrease)
            new_budget = max(current * (1 - decrease), self.min_budget)
            action = 'decrease'
            reasoning = f"ROAS below 1.0. Reducing budget by {decrease*100:.0f}%"
            confidence = 0.75
            
        else:
            # Maintain
            new_budget = current
            action = 'maintain'
            reasoning = "Performance within acceptable range"
            confidence = 0.7
        
        return BudgetDecision(
            campaign_id=campaign.campaign_id,
            current_budget=current,
            recommended_budget=new_budget,
            action=action,
            predicted_roas=predicted_roas,
            confidence=confidence,
            reasoning=reasoning
        )


# Singleton service
_optimizer_service: Optional[ROASOptimizerService] = None


def get_roas_optimizer() -> ROASOptimizerService:
    """Get or create the ROAS optimizer service."""
    global _optimizer_service
    if _optimizer_service is None:
        _optimizer_service = ROASOptimizerService()
    return _optimizer_service
