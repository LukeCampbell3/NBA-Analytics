#!/usr/bin/env python3
"""
Hybrid Spike-Driver MoE: Best of Both Worlds
- Baseline+delta architecture (prevents mean collapse)
- Spike drivers + tail weighting (eruption awareness)  
- MoE routing with spike experts (specialized capacity)
- Proper uncertainty quantification
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

@tf.keras.utils.register_keras_serializable()
class AddBaseline(tf.keras.layers.Layer):
    """
    Hard architectural constraint: forces delta prediction.
    Adds baseline to delta to produce final mean.
    This makes delta prediction a structural requirement, not just a loss convention.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        delta, baseline = inputs
        return delta + baseline
    
    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class ConditionalSpikeOutput(Layer):
    """Conditional 2-component spike mixture: Student-t for normal + spike-gated Student-t for spikes"""
    
    def __init__(self, use_probabilistic=True, min_scale=0.1, max_scale_pts=8.0, max_scale_trb=4.0, max_scale_ast=3.5, **kwargs):
        super().__init__(**kwargs)
        self.use_probabilistic = use_probabilistic
        self.min_scale = min_scale
        self.max_scale_pts = max_scale_pts
        self.max_scale_trb = max_scale_trb
        self.max_scale_ast = max_scale_ast
        self.k = 3  # PTS, TRB, AST
        
    def call(self, inputs):
        delta_output, baseline, spike_indicators = inputs
        
        if self.use_probabilistic:
            # Parse outputs: means, scales, spike_deltas, spike_scales
            means = delta_output[:, :self.k]  # Normal component means
            scales = delta_output[:, self.k:2*self.k]  # Normal component scales
            spike_deltas = delta_output[:, 2*self.k:3*self.k]  # Spike component delta means
            spike_scales = delta_output[:, 3*self.k:4*self.k]  # Spike component scales
            
            # Baseline-anchored normal component
            normal_means = baseline + means
            
            # Spike component: baseline + normal_delta + spike_delta
            spike_means = normal_means + spike_deltas
            
            # Apply prediction bounds to both components
            min_bounds = tf.constant([1.0, 0.0, 0.0], dtype=tf.float32)
            max_bounds = tf.constant([70.0, 25.0, 20.0], dtype=tf.float32)
            
            # Normal component bounds
            normal_normalized = tf.sigmoid(normal_means / 10.0)
            normal_means_bounded = min_bounds + (max_bounds - min_bounds) * normal_normalized
            
            # Spike component bounds (allow higher values)
            spike_normalized = tf.sigmoid(spike_means / 15.0)  # Wider range for spikes
            spike_means_bounded = min_bounds + (max_bounds - min_bounds) * spike_normalized
            
            # Smooth bounded scales for both components
            min_scales = tf.constant([self.min_scale] * self.k, dtype=tf.float32)
            max_scales = tf.constant([self.max_scale_pts, self.max_scale_trb, self.max_scale_ast], dtype=tf.float32)
            
            normal_scales_bounded = min_scales + (max_scales - min_scales) * tf.sigmoid(scales)
            spike_scales_bounded = min_scales + (max_scales - min_scales) * tf.sigmoid(spike_scales)
            
            # Spike-aware uncertainty scaling
            spike_multipliers = 1.0 + 0.3 * spike_indicators
            normal_scales_final = normal_scales_bounded * spike_multipliers
            spike_scales_final = spike_scales_bounded * (1.0 + spike_indicators)  # Extra scaling for spike component
            
            # Return: [normal_means, normal_scales, spike_means, spike_scales, spike_indicators]
            return tf.concat([
                normal_means_bounded, normal_scales_final,
                spike_means_bounded, spike_scales_final,
                spike_indicators
            ], axis=-1)
        else:
            # For non-probabilistic, just return normal component
            raw_means = baseline + delta_output
            min_bounds = tf.constant([1.0, 0.0, 0.0], dtype=tf.float32)
            max_bounds = tf.constant([70.0, 25.0, 20.0], dtype=tf.float32)
            normalized = tf.sigmoid(raw_means / 10.0)
            means = min_bounds + (max_bounds - min_bounds) * normalized
            return means
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "use_probabilistic": self.use_probabilistic,
            "min_scale": self.min_scale,
            "max_scale_pts": self.max_scale_pts,
            "max_scale_trb": self.max_scale_trb,
            "max_scale_ast": self.max_scale_ast
        })
        return config

class HybridSpikeMoETrainer:
    """Hybrid MoE combining baseline+delta with spike drivers and ensemble uncertainty"""
    
    def __init__(self, ensemble_size=3):
        self.models = []  # Store multiple models for ensemble
        self.ensemble_size = ensemble_size
        self.scaler_x = None
        self.feature_columns = None
        self.target_columns = ["PTS", "TRB", "AST"]
        self.player_mapping = None
        self.team_mapping = None
        self.opponent_mapping = None
        
        self.config = {
            "seq_len": 10,
            "d_model": 256,  # FIXED: Double capacity for better predictions
            "n_heads": 8,  # FIXED: More attention heads for complexity
            "n_layers": 4,  # FIXED: More layers for complexity
            "dropout": 0.1,
            "batch_size": 32,  # FIXED: Smaller batches for better gradients
            "epochs": 50,  # FIXED: Focus on quality over quantity
            "lr": 0.003,  # FIXED: Higher learning rate for better convergence
            "patience": 15,
            "expert_dim": 256,  # FIXED: Double expert capacity
            "num_experts": 8,  # BALANCED: More regular experts for normal games
            "num_spike_experts": 3,  # BALANCED: Adequate spike experts without dominance
            "use_probabilistic": True,
            "min_scale": 0.1,
            "max_scale_pts": 8.0,
            "max_scale_trb": 4.0,
            "max_scale_ast": 3.5,
            "spike_thresholds": {"PTS": 35.0, "TRB": 11.0, "AST": 9.0},
            "spike_loss_weights": {"PTS": 2.0, "TRB": 1.8, "AST": 1.5},  # >1 for tail emphasis
            
            # CRITICAL FIXES FOR ROUTING ISSUES + EXPERT DIVERSITY
            "load_balance_weight": 0.08,  # INCREASED: Much stronger load balancing for expert diversity
            "router_z_loss_weight": 0.02,  # INCREASED: Stronger entropy encouragement for diversity
            "gate_keepalive_weight": 0.01,  # INCREASED: Prevent expert dropout
            "router_temperature": 2.0,  # INCREASED: Much higher temperature for diverse routing
            "confidence_threshold": 0.45,  # REDUCED: Allow less confident but more diverse decisions
            "entropy_target": 1.8,  # INCREASED: Target higher entropy for more expert diversity
            "spike_routing_strength": 1.5,  # BALANCED: Moderate bias based on actual indicators
            "gradient_flow_fix": False,  # FIXED: Disable to allow natural routing
            
            # PREDICTION QUALITY FIXES
            "delta_l2_weight": 0.00001,  # FIXED: Reduce 100x to prevent over-regularization
            "delta_centering_weight": 0.0,  # REMOVED: Hard centering was too aggressive
            "use_student_t": True,  # Use Student-t likelihood for heavy tails
            "student_t_df": 1.5,  # FIXED: Heavier tails for more variance
            "ensemble_diversity": True,  # Make ensemble members different
            "spike_routing_bias": 0.01,  # FIXED: Much weaker bias to prevent collapse
            "spike_expert_capacity": 512,  # FIXED: Double spike expert capacity
            
            # ENHANCED TARGETED IMPROVEMENTS
            # Improvement 1: EXTREME variance encouragement (MUCH STRONGER)
            "variance_encouragement_weight": 1.0,  # FIXED: Triple strength for better variance
            "target_variance_ratios": [0.4, 0.5, 0.3],  # FIXED: More achievable targets
            "variance_loss_type": "exponential_penalty",  # Use exponential penalty for stronger effect
            "severe_suppression_penalty": 0.5,  # FIXED: Much stronger penalty for ratios < 0.1
            
            # Improvement 2: Enhanced spike handling
            "spike_detection_sensitivity": 0.8,  # FIXED: Less sensitive to prevent always-spike
            "spike_expert_routing_boost": 0.5,  # INCREASED: Much stronger routing to spike experts
            "spike_context_features": True,  # Use additional context for spike detection
            "spike_frequency_target": 0.08,  # FIXED: Very conservative spike routing target
            "spike_frequency_weight": 0.05,  # ENHANCED: Stronger frequency matching
            
            # Improvement 3: Uncertainty calibration improvements
            "uncertainty_calibration_weight": 0.02,  # Explicit uncertainty calibration loss
            "adaptive_uncertainty": True,  # Make uncertainty adaptive to prediction confidence
            "uncertainty_target_correlation": 0.15,  # Target σ-error correlation
            
            # Improvement 4: Ensemble diversity encouragement (NEW)
            "ensemble_diversity_encouragement": True,  # Encourage models to make different predictions
            "diversity_loss_weights": [0.10, 0.05, 0.02],  # Diversity encouragement weights per strategy
            
            # NEW: Enhanced expert diversity and utilization
            "expert_utilization_balancing": True,  # ENABLED: Balance expert utilization
            "multi_threshold_routing": True,  # NEW: Multi-level spike routing
            "enhanced_context_detection": True,  # NEW: Enhanced context-aware detection
            "regime_separation_weight": 0.005,  # REDUCED: Light regime separation to allow diversity
            "expert_specialization_weight": 0.001,  # REDUCED: Very light specialization to encourage diversity
            "expert_dropout_prevention": True,  # ENABLED: Prevent expert dropout for diversity
            "min_expert_usage": 0.05,  # INCREASED: Higher minimum expert usage threshold
            "underused_expert_boost": 0.15,  # INCREASED: Stronger boost for underused experts
            "expert_diversity_weight": 0.03,  # NEW: Explicit expert diversity encouragement
            "max_expert_concentration": 0.4,  # NEW: Prevent any expert from being used >40% of time
            "high_spike_threshold": 0.7,  # NEW: High confidence spike threshold
            "regime_specific_bias": True,  # NEW: Additional bias for high-confidence spikes
                    }
    
    def create_hybrid_features(self, df):
        """Create features optimized for baseline+delta architecture"""
        
        print("🎯 Creating hybrid spike driver features...")
        
        # USER REQUESTED: Add missing critical features first
        df = self._add_missing_critical_features(df)
        
        # Keep rolling averages in REAL UNITS (unscaled) - critical for baseline+delta
        baseline_features = [
            "PTS_rolling_avg", "TRB_rolling_avg", "AST_rolling_avg"
        ]
        
        # Categorical features (will be embedded)
        categorical_features = [
            "Player_ID_mapped", "Team_ID_mapped", "Opponent_ID_mapped"
        ]
        
        # Context features (will be scaled) - EXPANDED with user-requested features
        context_features = [
            "BackToBack", "Home", "High_Volume_Flag", "Month_sin", "Month_cos",
            "FG%_rolling_avg", "USG%_rolling_avg", "GmSc_rolling_avg",
            # USER REQUESTED: Minutes & role features
            "MP_rolling_avg", "Starter_Flag", "MP_last_game_change",
            # USER REQUESTED: Usage/involvement features
            "FGA_rate", "FTA_rate", "USG_proxy",
            # USER REQUESTED: Context features
            "Rest_Days", "Opponent_DRtg_rolling",
            # USER REQUESTED: Pace estimate
            "Pace_estimate"
        ]
        
        # Delta features (recent vs baseline)
        delta_features = [
            "PTS_delta", "TRB_delta", "AST_delta",
            "PTS_rolling_std", "TRB_rolling_std", "AST_rolling_std"
        ]
        
        # Spike driver features (the eruption indicators)
        spike_features = [
            "MP_trend", "High_MP_Flag", "FGA_trend", "AST_trend", 
            "AST_variance", "USG_AST_ratio_trend", "High_Playmaker_Flag"
        ]
        
        # Initialize spike_features as instance variable before calling _create_spike_drivers
        self.spike_features = spike_features.copy()
        
        # Create spike drivers
        self._create_spike_drivers(df)
        
        # Ensure all computed features exist
        for stat in self.target_columns:
            if f"{stat}_delta" not in df.columns:
                if f"{stat}_rolling_avg" in df.columns and stat in df.columns:
                    df[f"{stat}_delta"] = df[stat] - df[f"{stat}_rolling_avg"]
                else:
                    df[f"{stat}_delta"] = 0.0
            
            if f"{stat}_rolling_std" not in df.columns:
                if stat in df.columns:
                    df[f"{stat}_rolling_std"] = df.groupby('Player_Name')[stat].transform(
                        lambda x: x.rolling(5, min_periods=2).std().fillna(0)
                    )
                else:
                    df[f"{stat}_rolling_std"] = 0.0
        
        # Define feature groups for different processing
        self.baseline_features = baseline_features
        self.categorical_features = categorical_features
        self.context_features = context_features
        self.delta_features = delta_features
        # self.spike_features already set above and potentially updated by _create_spike_drivers
        
        # All features for model input (baselines kept separate)
        self.feature_columns = categorical_features + context_features + delta_features + spike_features
        
        # CRITICAL: Filter feature_columns to only include columns that actually exist in df
        existing_features = [col for col in self.feature_columns if col in df.columns]
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        
        if missing_features:
            print(f"⚠️  Warning: {len(missing_features)} features not found in data, creating with default values:")
            for col in missing_features:
                print(f"    - {col}")
                df[col] = 0.0  # Default to 0 for missing features
        
        self.feature_columns = existing_features + missing_features  # Use all features (existing + created)
        
        print(f"✅ Created hybrid feature set:")
        print(f"  Baseline features (unscaled): {len(baseline_features)}")
        print(f"  Model features (processed): {len(self.feature_columns)}")
        print(f"  Total features: {len(baseline_features) + len(self.feature_columns)}")
        
        return df
    
    def _add_missing_critical_features(self, df):
        """
        USER REQUESTED: Add missing 'why did performance change?' features
        These typically produce immediate R²/trajectory gains
        """
        print("🔧 Adding user-requested critical features...")
        
        # 1. Minutes & role features
        if "MP" in df.columns:
            # Rolling average minutes
            df["MP_rolling_avg"] = df.groupby('Player_Name')["MP"].transform(
                lambda x: x.rolling(5, min_periods=2).mean().fillna(x.mean())
            )
            
            # Starter flag (>= 25 minutes typically means starter)
            df["Starter_Flag"] = (df["MP"] >= 25.0).astype(int)
            
            # Last game minutes change
            df["MP_last_game_change"] = df.groupby('Player_Name')["MP"].transform(
                lambda x: x.diff().fillna(0)
            )
        else:
            df["MP_rolling_avg"] = 30.0  # Default assumption
            df["Starter_Flag"] = 1
            df["MP_last_game_change"] = 0.0
        
        # 2. Usage/involvement features
        if "FGA" in df.columns and "MP" in df.columns:
            # FGA rate (per 36 minutes)
            df["FGA_rate"] = (df["FGA"] / (df["MP"] + 1)) * 36
        else:
            df["FGA_rate"] = 0.0
        
        if "FTA" in df.columns and "MP" in df.columns:
            # FTA rate (per 36 minutes)
            df["FTA_rate"] = (df["FTA"] / (df["MP"] + 1)) * 36
        else:
            df["FTA_rate"] = 0.0
        
        if "USG%" in df.columns:
            # USG proxy (already have USG%, just copy it)
            df["USG_proxy"] = df["USG%"]
        elif "FGA" in df.columns and "FTA" in df.columns and "TOV" in df.columns:
            # Approximate USG% if not available
            # USG% ≈ 100 * (FGA + 0.44*FTA + TOV) / (MP * TeamPace/48)
            # Simplified: just use FGA + 0.44*FTA + TOV as proxy
            df["USG_proxy"] = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]
        else:
            df["USG_proxy"] = 0.0
        
        # 3. Rest days (from Date column)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            df["Rest_Days"] = df.groupby('Player_Name')["Date"].transform(
                lambda x: x.diff().dt.days.fillna(2)  # Default 2 days rest
            ).clip(0, 10)  # Cap at 10 days
        else:
            df["Rest_Days"] = 2.0  # Default assumption
        
        # BackToBack is already in the data, but ensure it exists
        if "BackToBack" not in df.columns:
            if "Rest_Days" in df.columns:
                df["BackToBack"] = (df["Rest_Days"] <= 1).astype(int)
            else:
                df["BackToBack"] = 0
        
        # 4. Opponent defensive strength
        if "DRTG" in df.columns:
            # Use opponent's defensive rating (lower is better defense)
            # Rolling average of opponent DRtg
            df["Opponent_DRtg_rolling"] = df.groupby('Opponent_ID')["DRTG"].transform(
                lambda x: x.rolling(10, min_periods=3).mean().fillna(x.mean())
            )
        elif "oppDfRtg_3" in df.columns:
            # Use existing opponent defensive rating feature
            df["Opponent_DRtg_rolling"] = df["oppDfRtg_3"]
        else:
            df["Opponent_DRtg_rolling"] = 110.0  # League average
        
        # 5. Pace estimate
        if "GmSc" in df.columns and "MP" in df.columns:
            # Rough pace proxy: higher GmSc/MP suggests faster pace
            df["Pace_estimate"] = (df["GmSc"] / (df["MP"] + 1)) * 48
        else:
            df["Pace_estimate"] = 100.0  # League average pace
        
        # Home/Away already exists as "Home" column
        
        print("✅ Added critical features:")
        print("  - Minutes & role: MP_rolling_avg, Starter_Flag, MP_last_game_change")
        print("  - Usage/involvement: FGA_rate, FTA_rate, USG_proxy")
        print("  - Context: Rest_Days, BackToBack (existing), Home (existing)")
        print("  - Opponent: Opponent_DRtg_rolling")
        print("  - Pace: Pace_estimate")
        
        return df
    
    def _create_spike_drivers(self, df):
        """Create enhanced spike driver features with better sensitivity"""
        
        # MP trend and flag
        if "MP" in df.columns:
            df["MP_trend"] = df.groupby('Player_Name')["MP"].transform(
                lambda x: x.rolling(3, min_periods=2).apply(
                    lambda y: (y.iloc[-1] - y.iloc[0]) / len(y) if len(y) > 1 else 0
                ).fillna(0)
            )
            df["High_MP_Flag"] = (df["MP"] >= 28.0).astype(int)
        else:
            df["MP_trend"] = 0.0
            df["High_MP_Flag"] = 0
        
        # FGA trend
        if "FGA" in df.columns:
            df["FGA_trend"] = df.groupby('Player_Name')["FGA"].transform(
                lambda x: x.rolling(3, min_periods=2).apply(
                    lambda y: (y.iloc[-1] - y.iloc[0]) / len(y) if len(y) > 1 else 0
                ).fillna(0)
            )
        else:
            df["FGA_trend"] = 0.0
        
        # AST trend and variance
        if "AST" in df.columns:
            df["AST_trend"] = df.groupby('Player_Name')["AST"].transform(
                lambda x: x.rolling(3, min_periods=2).apply(
                    lambda y: (y.iloc[-1] - y.iloc[0]) / len(y) if len(y) > 1 else 0
                ).fillna(0)
            )
            df["AST_variance"] = df.groupby('Player_Name')["AST"].transform(
                lambda x: x.rolling(5, min_periods=3).var().fillna(0)
            )
        else:
            df["AST_trend"] = 0.0
            df["AST_variance"] = 0.0
        
        # USG-AST ratio trend
        if "USG%" in df.columns and "AST" in df.columns:
            df["USG_AST_ratio"] = df["USG%"] / (df["AST"] + 1)
            df["USG_AST_ratio_trend"] = df.groupby('Player_Name')["USG_AST_ratio"].transform(
                lambda x: x.rolling(3, min_periods=2).apply(
                    lambda y: (y.iloc[-1] - y.iloc[0]) / len(y) if len(y) > 1 else 0
                ).fillna(0)
            )
        else:
            df["USG_AST_ratio_trend"] = 0.0
        
        # High playmaker flag
        if "AST" in df.columns and "TOV" in df.columns:
            df["AST_TOV_ratio"] = df["AST"] / (df["TOV"] + 1)
            df["High_Playmaker_Flag"] = (df["AST_TOV_ratio"] >= 2.0).astype(int)
        elif "AST" in df.columns:
            df["High_Playmaker_Flag"] = (df["AST"] >= 6.0).astype(int)
        else:
            df["High_Playmaker_Flag"] = 0
        
        # NEW: Enhanced spike context features
        if self.config.get("spike_context_features", False):
            # Recent performance momentum
            for stat in self.target_columns:
                if stat in df.columns:
                    # Performance momentum (last 3 games vs season average)
                    df[f"{stat}_momentum"] = df.groupby('Player_Name')[stat].transform(
                        lambda x: x.rolling(3, min_periods=2).mean() - x.expanding().mean()
                    ).fillna(0)
                    
                    # Performance volatility (recent standard deviation)
                    df[f"{stat}_volatility"] = df.groupby('Player_Name')[stat].transform(
                        lambda x: x.rolling(5, min_periods=3).std()
                    ).fillna(0)
            
            # Game context indicators
            if "Home" in df.columns and "BackToBack" in df.columns:
                # Favorable context (home + rested)
                df["Favorable_Context"] = ((df["Home"] == 1) & (df["BackToBack"] == 0)).astype(int)
                
            # Update feature lists to include new features
            momentum_features = [f"{stat}_momentum" for stat in self.target_columns]
            volatility_features = [f"{stat}_volatility" for stat in self.target_columns]
            context_features = ["Favorable_Context"] if "Favorable_Context" in df.columns else []
            
            # Add to spike features
            self.spike_features.extend(momentum_features + volatility_features + context_features)
    
    def prepare_data(self):
        """Load and prepare training data with hybrid architecture"""
        
        print("📊 Loading training data...")
        data_dir = Path("Data")
        
        training_players = [
            "Stephen_Curry", "LeBron_James", "Giannis_Antetokounmpo", 
            "Luka_Doncic", "Jayson_Tatum", "Kevin_Durant", "Nikola_Jokic",
            "Joel_Embiid", "Damian_Lillard", "Jimmy_Butler"
        ]
        
        all_data = []
        for player_dir in data_dir.iterdir():
            if not player_dir.is_dir():
                continue
            
            if player_dir.name in training_players:
                for csv_file in player_dir.glob("*_processed.csv"):
                    try:
                        df = pd.read_csv(csv_file)
                        df["Player_Name"] = player_dir.name
                        
                        # Extract Team_ID from one-hot encoded TM_ columns
                        tm_cols = [col for col in df.columns if col.startswith('TM_')]
                        if tm_cols and len(tm_cols) > 0:
                            # Find which team column is 1 for each row
                            df['Team_ID'] = df[tm_cols].idxmax(axis=1).str.replace('TM_', '')
                        else:
                            # Fallback: use a default team ID
                            df['Team_ID'] = 'UNK'
                        
                        # Extract Opponent_ID from one-hot encoded OPP_ columns
                        opp_cols = [col for col in df.columns if col.startswith('OPP_')]
                        if opp_cols and len(opp_cols) > 0:
                            # Find which opponent column is 1 for each row
                            df['Opponent_ID'] = df[opp_cols].idxmax(axis=1).str.replace('OPP_', '')
                        else:
                            # Fallback: use a default opponent ID
                            df['Opponent_ID'] = 'UNK'
                        
                        # Verify columns were created
                        if 'Team_ID' not in df.columns or 'Opponent_ID' not in df.columns:
                            print(f"  ⚠️  Warning: Missing Team_ID or Opponent_ID in {csv_file.name}")
                            df['Team_ID'] = df.get('Team_ID', 'UNK')
                            df['Opponent_ID'] = df.get('Opponent_ID', 'UNK')
                        
                        all_data.append(df)
                        print(f"✓ Loaded {len(df)} games for {player_dir.name}")
                    except Exception as e:
                        print(f"Error loading {csv_file}: {e}")
                        import traceback
                        traceback.print_exc()
        
        if not all_data:
            raise ValueError("No training data loaded!")
        
        df = pd.concat(all_data, ignore_index=True)
        print(f"Total training games: {len(df):,}")
        
        # Verify Team_ID and Opponent_ID exist after concatenation
        if 'Team_ID' not in df.columns:
            print("❌ Team_ID column missing after concatenation!")
            print(f"Available columns: {df.columns.tolist()[:20]}")
            raise ValueError("Team_ID column not found in concatenated dataframe")
        
        if 'Opponent_ID' not in df.columns:
            print("❌ Opponent_ID column missing after concatenation!")
            raise ValueError("Opponent_ID column not found in concatenated dataframe")
        print(f"Total training games: {len(df):,}")
        
        # Create ID mappings
        self.player_mapping = {name: idx for idx, name in enumerate(df['Player_Name'].unique())}
        self.team_mapping = {team_id: idx for idx, team_id in enumerate(sorted(df['Team_ID'].unique()))}
        self.opponent_mapping = {opp_id: idx for idx, opp_id in enumerate(sorted(df['Opponent_ID'].unique()))}
        
        df['Player_ID_mapped'] = df['Player_Name'].map(self.player_mapping)
        df['Team_ID_mapped'] = df['Team_ID'].map(self.team_mapping)
        df['Opponent_ID_mapped'] = df['Opponent_ID'].map(self.opponent_mapping)
        
        # Create hybrid features
        df = self.create_hybrid_features(df)
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values for model features (baselines handled separately)
        for col in self.feature_columns:
            if col in df.columns:
                if col in self.categorical_features:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Fill baseline features (keep in real units)
        for col in self.baseline_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Create sequences
        print("Creating training sequences...")
        X, baselines, y = self.create_sequences(df)
        
        print(f"Training data shape: X={X.shape}, baselines={baselines.shape}, y={y.shape}")
        
        return X, baselines, y, df
    
    def create_sequences(self, df):
        """Create sequences with separate baseline handling"""
        
        X, baselines, y = [], [], []
        seq_len = self.config["seq_len"]
        
        for player_name in df['Player_Name'].unique():
            player_df = df[df['Player_Name'] == player_name].copy()
            player_df = player_df.sort_values('Game_Index').reset_index(drop=True)
            
            if len(player_df) < seq_len + 1:
                continue
            
            for i in range(seq_len, len(player_df)):
                # Skip DNP games
                if player_df.iloc[i][self.target_columns].sum() < 2:
                    continue
                
                # Get sequence for model features
                sequence = player_df.iloc[i-seq_len:i][self.feature_columns].values
                
                # Get baseline from last timestep (in real units)
                baseline = player_df.iloc[i-1][self.baseline_features].values
                
                # Get target
                target = player_df.iloc[i][self.target_columns].values
                
                # Convert to float and check for missing values
                sequence = np.array(sequence, dtype=np.float32)
                baseline = np.array(baseline, dtype=np.float32)
                target = np.array(target, dtype=np.float32)
                
                if np.isnan(sequence).any() or np.isnan(baseline).any() or np.isnan(target).any():
                    continue
                
                X.append(sequence)
                baselines.append(baseline)
                y.append(target)
        
        return (np.array(X, dtype=np.float32), 
                np.array(baselines, dtype=np.float32), 
                np.array(y, dtype=np.float32))
    
    def build_model(self):
        """Build hybrid MoE with baseline+delta architecture"""
        
        print("🏗️ Building Hybrid Spike-Driver MoE...")
        
        # Input layers
        sequence_input = Input(shape=(self.config["seq_len"], len(self.feature_columns)), name="sequence_input")
        baseline_input = Input(shape=(len(self.baseline_features),), name="baseline_input")
        
        # Extract categorical features for embedding
        player_ids = Lambda(lambda x: tf.cast(x[:, :, 0], tf.int32), name="player_ids")(sequence_input)
        team_ids = Lambda(lambda x: tf.cast(x[:, :, 1], tf.int32), name="team_ids")(sequence_input)
        opponent_ids = Lambda(lambda x: tf.cast(x[:, :, 2], tf.int32), name="opponent_ids")(sequence_input)
        
        # Embeddings
        player_embed = Embedding(len(self.player_mapping), 16, name="player_embed")(player_ids)
        team_embed = Embedding(len(self.team_mapping), 8, name="team_embed")(team_ids)
        opponent_embed = Embedding(len(self.opponent_mapping), 8, name="opponent_embed")(opponent_ids)
        
        # Numeric features (skip first 3 categorical)
        numeric_features = Lambda(lambda x: x[:, :, 3:], name="numeric_features")(sequence_input)
        
        # Combine features
        combined = Concatenate(axis=-1, name="combined_features")([
            player_embed, team_embed, opponent_embed, numeric_features
        ])
        
        # Project to model dimension
        x = Dense(self.config["d_model"], name="input_projection")(combined)
        
        # Transformer layers
        for i in range(self.config["n_layers"]):
            # Multi-head attention
            attn = MultiHeadAttention(
                num_heads=self.config["n_heads"],
                key_dim=self.config["d_model"] // self.config["n_heads"],
                name=f"attention_{i}"
            )(x, x)
            
            x = Add(name=f"add_attn_{i}")([x, attn])
            x = LayerNormalization(name=f"norm_attn_{i}")(x)
            
            # Feed forward
            ff = Dense(self.config["d_model"] * 2, activation="relu", name=f"ff1_{i}")(x)
            ff = Dropout(self.config["dropout"], name=f"dropout_ff_{i}")(ff)
            ff = Dense(self.config["d_model"], name=f"ff2_{i}")(ff)
            
            x = Add(name=f"add_ff_{i}")([x, ff])
            x = LayerNormalization(name=f"norm_ff_{i}")(x)
        
        # Use full sequence for routing (not just last timestep)
        sequence_repr = GlobalAveragePooling1D(name="sequence_pooling")(x)
        
        # Fix #2: Condition router/experts on baseline (concat baseline into sequence_repr)
        sequence_repr_with_baseline = Concatenate(name="sequence_baseline_concat")([sequence_repr, baseline_input])
        
        # Fix #1: Per-target spike indicators (3-dimensional for PTS/TRB/AST)
        spike_features_last = Lambda(lambda x: x[:, -1, -len(self.spike_features):], name="spike_features")(sequence_input)
        spike_indicators = Dense(len(self.target_columns), activation="sigmoid", name="per_target_spike_indicators")(spike_features_last)
        
        # Enhanced spike routing with temperature scaling and confidence targeting
        total_experts = self.config["num_experts"] + self.config["num_spike_experts"]
        router_logits = Dense(total_experts, name="router")(sequence_repr_with_baseline)
        
        # EMERGENCY FIX: Scale down logits to reduce extreme confidence
        router_logits = router_logits * 0.1  # Reduce magnitude 10x
        
        # Apply temperature scaling for sharper decisions
        temperature = self.config.get("router_temperature", 1.0)
        router_logits_scaled = router_logits / temperature
        
        # Apply spike routing bias for better spike expert utilization
        spike_expert_mask = tf.concat([
            tf.zeros([tf.shape(router_logits_scaled)[0], self.config["num_experts"]]),
            tf.ones([tf.shape(router_logits_scaled)[0], self.config["num_spike_experts"]])
        ], axis=1)
        avg_spike_score = tf.reduce_mean(spike_indicators, axis=1, keepdims=True)  # Average across PTS/TRB/AST
        
        # Apply much stronger routing boost for spike experts
        routing_strength = self.config.get("spike_routing_strength", 1.0)
        spike_routing_bias = avg_spike_score * spike_expert_mask * routing_strength
        
        # Adjust router probabilities with spike bias
        adjusted_router_logits = router_logits_scaled + spike_routing_bias
        router_probs = Softmax(name="router_probs")(adjusted_router_logits)
        
        # Add gradient flow fix: ensure all experts get some gradient flow
        if self.config.get("gradient_flow_fix", False):
            # Add small uniform distribution to prevent complete expert dropout
            uniform_dist = tf.ones_like(router_probs) / total_experts
            router_probs = 0.97 * router_probs + 0.03 * uniform_dist  # Minimal intervention for natural specialization
        
        # Expert networks (regular + spike experts) - using baseline-conditioned representation
        expert_outputs = []
        for i in range(total_experts):
            expert_type = "spike" if i >= self.config["num_experts"] else "regular"
            # Fix #3: Enhanced spike expert specialization - larger capacity for spike experts
            expert_dim = self.config["spike_expert_capacity"] if expert_type == "spike" else self.config["expert_dim"]
            expert = Dense(expert_dim, activation="relu", name=f"expert_{expert_type}_{i}_1")(sequence_repr_with_baseline)
            expert = Dropout(self.config["dropout"], name=f"expert_{expert_type}_{i}_dropout")(expert)
            
            if self.config["use_probabilistic"]:
                expert_out = Dense(len(self.target_columns) * 4, name=f"expert_{expert_type}_{i}_out")(expert)
            else:
                expert_out = Dense(len(self.target_columns), name=f"expert_{expert_type}_{i}_out")(expert)
            
            expert_outputs.append(expert_out)
        
        # Combine expert outputs with utilization balancing
        expert_stack = tf.stack(expert_outputs, axis=1)
        router_probs_expanded = tf.expand_dims(router_probs, axis=-1)
        
        # Add expert utilization balancing for better gradient flow
        if self.config.get("expert_utilization_balancing", False):
            # Compute expert usage statistics
            expert_usage = tf.reduce_mean(router_probs, axis=0)
            # Apply gentle balancing to underused experts (configurable)
            min_usage = self.config.get("min_expert_usage", 0.0)
            underused_boost_weight = self.config.get("underused_expert_boost", 0.1)
            underused_boost = tf.maximum(0.0, min_usage - expert_usage) * underused_boost_weight
            
            # Apply boost to router probabilities
            router_probs_balanced = router_probs + tf.expand_dims(underused_boost, 0)
            router_probs_balanced = router_probs_balanced / tf.reduce_sum(router_probs_balanced, axis=1, keepdims=True)
            
            router_probs_expanded = tf.expand_dims(router_probs_balanced, axis=-1)
        
        delta_output = tf.reduce_sum(expert_stack * router_probs_expanded, axis=1)
        
        # Apply hybrid output layer
        final_output = ConditionalSpikeOutput(
            use_probabilistic=self.config["use_probabilistic"],
            min_scale=self.config["min_scale"],
            max_scale_pts=self.config["max_scale_pts"],
            max_scale_trb=self.config["max_scale_trb"],
            max_scale_ast=self.config["max_scale_ast"],
            name="conditional_spike_output"
        )([delta_output, baseline_input, spike_indicators])
        
        # Create model
        model = Model(inputs=[sequence_input, baseline_input], outputs=final_output, name="HybridSpikeMoE")
        
        # Add enhanced MoE regularization losses to the model
        # Load balance loss: encourage equal expert usage (REDUCED weight for specialization)
        router_probs_mean = tf.reduce_mean(router_probs, axis=0)
        load_balance_loss_val = tf.reduce_sum(tf.square(router_probs_mean - 1.0/total_experts))
        model.add_loss(self.config["load_balance_weight"] * load_balance_loss_val)
        
        # Expert diversity penalty: discourage expert collapse and over-concentration
        if self.config.get("expert_diversity_weight", 0) > 0:
            min_usage = self.config.get("min_expert_usage", 0.0)
            max_usage = self.config.get("max_expert_concentration", 1.0)
            underuse_penalty = tf.nn.relu(min_usage - router_probs_mean)
            overuse_penalty = tf.nn.relu(router_probs_mean - max_usage)
            diversity_penalty = tf.reduce_mean(tf.square(underuse_penalty) + tf.square(overuse_penalty))
            model.add_loss(self.config["expert_diversity_weight"] * diversity_penalty)
            model.add_metric(diversity_penalty, name='expert_diversity_penalty')
        
        # Router entropy: EMERGENCY FIX - Massively increase weight to force diversity
        router_entropy_val = -tf.reduce_mean(tf.reduce_sum(router_probs * tf.math.log(router_probs + 1e-8), axis=1))
        entropy_target = self.config.get("entropy_target", 1.0)
        
        # Only penalize when entropy is BELOW target
        entropy_deficit = tf.nn.relu(entropy_target - router_entropy_val)
        entropy_loss = tf.square(entropy_deficit)
        
        # EMERGENCY FIX: Increase weight from 0.001 to 2.0 (2000x increase!)
        entropy_weight = 2.0
        model.add_loss(entropy_weight * entropy_loss)
        
        # Confidence targeting: encourage high-confidence decisions
        confidence_threshold = self.config.get("confidence_threshold", 0.5)
        max_probs = tf.reduce_max(router_probs, axis=1)
        confidence_loss = tf.reduce_mean(tf.maximum(0.0, confidence_threshold - max_probs))
        model.add_loss(0.01 * confidence_loss)
        
        # Gate keepalive: prevent experts from being completely unused (REDUCED weight)
        gate_keepalive_val = tf.reduce_mean(tf.reduce_sum(tf.cast(router_probs > 0.01, tf.float32), axis=1))
        model.add_loss(self.config["gate_keepalive_weight"] * (-gate_keepalive_val))
        
        # Spike frequency targeting: ensure adequate spike expert usage
        spike_expert_usage = tf.reduce_mean(tf.reduce_sum(router_probs[:, self.config["num_experts"]:], axis=1))
        spike_frequency_target = self.config.get("spike_frequency_target", 0.15)
        spike_frequency_loss = tf.square(spike_expert_usage - spike_frequency_target)
        model.add_loss(self.config.get("spike_frequency_weight", 0.01) * spike_frequency_loss)
        
        # Regime separation loss: encourage different routing for different spike levels
        if self.config.get("regime_separation_weight", 0) > 0:
            # High spike samples should route more to spike experts
            high_spike_samples = tf.cast(avg_spike_score > 0.6, tf.float32)
            low_spike_samples = tf.cast(avg_spike_score < 0.3, tf.float32)
            
            high_spike_routing = tf.reduce_sum(router_probs[:, self.config["num_experts"]:], axis=1)
            low_spike_routing = tf.reduce_sum(router_probs[:, :self.config["num_experts"]], axis=1)
            
            # Encourage high spike samples to use spike experts more
            high_spike_separation = tf.reduce_mean(high_spike_samples * (0.7 - high_spike_routing))
            # Encourage low spike samples to use regular experts more  
            low_spike_separation = tf.reduce_mean(low_spike_samples * (0.7 - low_spike_routing))
            
            regime_separation_loss = tf.maximum(0.0, high_spike_separation) + tf.maximum(0.0, low_spike_separation)
            model.add_loss(self.config["regime_separation_weight"] * regime_separation_loss)
            
            model.add_metric(regime_separation_loss, name='regime_separation_loss')
        
        # Spike routing prior loss: encourage spike experts when spike indicators are high
        avg_spike_score = tf.reduce_mean(spike_indicators, axis=1, keepdims=True)  # Average across PTS/TRB/AST
        spike_expert_mask = tf.concat([
            tf.zeros([tf.shape(router_probs)[0], self.config["num_experts"]]),
            tf.ones([tf.shape(router_probs)[0], self.config["num_spike_experts"]])
        ], axis=1)
        # FIXED: Penalize excessive spike routing (prevent collapse)
        spike_expert_usage = tf.reduce_sum(router_probs * spike_expert_mask, axis=1)
        target_spike_usage = 0.15  # Target 15% spike routing
        spike_routing_loss_val = tf.reduce_mean(tf.square(spike_expert_usage - target_spike_usage))
        model.add_loss(self.config["spike_routing_bias"] * spike_routing_loss_val)
        
        # Expert specialization loss: encourage experts to specialize on different patterns
        if self.config.get("expert_specialization_weight", 0) > 0:
            # Encourage different experts to activate for different input patterns
            expert_activations = tf.reduce_mean(router_probs, axis=0)  # Average activation per expert
            specialization_loss = -tf.reduce_sum(expert_activations * tf.math.log(expert_activations + 1e-8))
            model.add_loss(self.config["expert_specialization_weight"] * (-specialization_loss))
        
        # Add metric tracking for these losses
        model.add_metric(load_balance_loss_val, name='load_balance_loss')
        model.add_metric(router_entropy_val, name='router_entropy') 
        model.add_metric(confidence_loss, name='confidence_loss')
        model.add_metric(spike_frequency_loss, name='spike_frequency_loss')
        model.add_metric(gate_keepalive_val, name='gate_keepalive')
        model.add_metric(spike_routing_loss_val, name='spike_routing_loss')
        model.add_metric(max_probs, name='max_routing_confidence')
        model.add_metric(spike_expert_usage, name='spike_expert_usage')
                # Add diagnostic metrics for debugging router collapse
        model.add_metric(tf.reduce_mean(tf.reduce_max(router_probs, axis=1)), name='avg_max_prob')
        model.add_metric(tf.reduce_mean(tf.cast(tf.argmax(router_probs, axis=1) >= self.config["num_experts"], tf.float32)), name='spike_routing_rate')
        model.add_metric(tf.reduce_mean(avg_spike_score), name='avg_spike_score')
        
        # Add expert usage distribution
        for i in range(total_experts):
            model.add_metric(tf.reduce_mean(router_probs[:, i]), name=f'expert_{i}_usage')
        
        return model
    
    def train(self):
        """Train ensemble of hybrid spike-driver models with meaningful diversity"""
        
        print(f"🚀 Training Ensemble of {self.ensemble_size} Hybrid Spike-Driver MoE Models")
        print("=" * 80)
        
        # Load and prepare data (same for all models)
        X_train_scaled, baselines_train, y_train, X_val_scaled, baselines_val, y_val = self._prepare_training_data()
        
        # Define optimized training strategies for balanced performance
        ensemble_strategies = [
            {
                "name": "confident_spike_aware",
                "spike_loss_weights": {"PTS": 1.4, "TRB": 1.3, "AST": 1.2},  # Moderate tail emphasis
                "delta_l2_weight": 0.00001,  # FIXED: Light L2 for stability
                "student_t_df": 1.2,  # FIXED: Heavy tails for spikes
                "dropout": 0.08,  # Moderate dropout
                "router_temperature": 1.0,  # FIXED: Balanced routing decisions
                "confidence_threshold": 0.55,  # ACHIEVABLE confidence target
                "entropy_target": 1.5,  # FIXED: Higher entropy for diversity
                "spike_routing_strength": 0.5,  # FIXED: Much weaker to prevent collapse
                # Optimized variance and spike focus
                "variance_encouragement_weight": 1.0,  # FIXED: Strong variance focus
                "uncertainty_calibration_weight": 0.02,  # FIXED: Balanced calibration
                "diversity_encouragement": 0.05,  # Moderate diversity
            },
            {
                "name": "balanced_performance", 
                "spike_loss_weights": {"PTS": 1.3, "TRB": 1.25, "AST": 1.15},  # Balanced tail emphasis
                "delta_l2_weight": 0.00002,  # FIXED: Light L2
                "student_t_df": 1.5,  # FIXED: Heavy tails
                "dropout": 0.10,
                "router_temperature": 1.1,  # FIXED: Less sharp routing
                "confidence_threshold": 0.52,  # Moderate confidence target
                "entropy_target": 1.6,  # FIXED: Higher entropy for diversity
                "spike_routing_strength": 0.3,  # FIXED: Weak to prevent collapse
                # Balanced focus
                "variance_encouragement_weight": 0.8,  # FIXED: Balanced variance focus
                "uncertainty_calibration_weight": 0.015,  # FIXED: Light calibration
                "diversity_encouragement": 0.04,  # Light diversity
            },
            {
                "name": "stable_intelligent_routing",
                "spike_loss_weights": {"PTS": 1.5, "TRB": 1.4, "AST": 1.3},  # Moderate tail emphasis
                "delta_l2_weight": 0.00003,  # FIXED: Light L2 for stability
                "student_t_df": 1.8,  # FIXED: Moderate tails
                "dropout": 0.12,
                "router_temperature": 1.2,  # FIXED: Stable and diverse
                "confidence_threshold": 0.50,  # Achievable confidence target
                "entropy_target": 1.7,  # FIXED: Highest entropy for diversity
                "spike_routing_strength": 0.2,  # FIXED: Very weak to prevent collapse
                # Stable focus
                "variance_encouragement_weight": 0.6,  # FIXED: Controlled variance focus
                "uncertainty_calibration_weight": 0.01,  # FIXED: Balanced calibration
                "diversity_encouragement": 0.03,  # Light diversity
            }
        ]
        
        # Train ensemble of models
        ensemble_histories = []
        
        for model_idx in range(self.ensemble_size):
            print(f"\n🔄 Training Model {model_idx + 1}/{self.ensemble_size}")
            strategy = ensemble_strategies[model_idx % len(ensemble_strategies)]
            print(f"📋 Strategy: {strategy['name']}")
            print("-" * 50)
            
            # Build model with different random seed for diversity
            tf.random.set_seed(42 + model_idx * 100)  # Different seed for each model
            model = self._build_model_with_strategy(strategy)
            
            def create_hybrid_loss_with_strategy(strategy):
                def hybrid_loss(y_true, y_pred):
                    if self.config["use_probabilistic"]:
                        # ConditionalSpikeOutput returns 15 dims:
                        # [normal_means(3), normal_scales(3), spike_means(3), spike_scales(3), spike_indicators(3)]
                        # We only use the first 6 (normal component) for loss calculation
                        means = y_pred[:, :len(self.target_columns)]
                        scales = y_pred[:, len(self.target_columns):2*len(self.target_columns)]
                        
                        # Student-t likelihood for heavy tails (working well)
                        if self.config.get("use_student_t", False):
                            df = strategy["student_t_df"]
                            # Student-t log-likelihood (heavy tails)
                            residuals = (y_true - means) / scales
                            log_likelihood = (
                                tf.math.lgamma((df + 1) / 2) - 
                                tf.math.lgamma(df / 2) - 
                                0.5 * tf.math.log(df * np.pi) - 
                                tf.math.log(scales) - 
                                ((df + 1) / 2) * tf.math.log(1 + tf.square(residuals) / df)
                            )
                            nll = -log_likelihood
                        else:
                            # Standard Gaussian NLL
                            nll = 0.5 * tf.math.log(2 * np.pi * tf.square(scales))
                            nll += 0.5 * tf.square((y_true - means) / scales)
                        
                        # Tail weighting with strategy-specific weights (working well)
                        spike_masks = []
                        for i, (stat, threshold) in enumerate(self.config["spike_thresholds"].items()):
                            mask = tf.cast(y_true[:, i] >= threshold, tf.float32)
                            weight = strategy["spike_loss_weights"][stat]
                            spike_masks.append(mask * weight + (1 - mask))
                        
                        spike_weights = tf.stack(spike_masks, axis=1)
                        nll = nll * spike_weights
                        
                        # IMPROVEMENT 1: EXTREME variance encouragement with exponential penalty
                        variance_loss = 0.0
                        if self.config.get("variance_encouragement_weight", 0) > 0:
                            pred_vars = tf.math.reduce_variance(means, axis=0)
                            target_vars = tf.math.reduce_variance(y_true, axis=0)
                            
                            # Use ratio-based penalty with exponential scaling
                            target_ratios = tf.constant(self.config.get("target_variance_ratios", [0.25, 0.30, 0.20]), dtype=tf.float32)
                            actual_ratios = pred_vars / (target_vars + 1e-6)
                            
                            if self.config.get("variance_loss_type") == "exponential_penalty":
                                # Exponential penalty for stronger effect
                                ratio_deficit = tf.maximum(0.0, target_ratios - actual_ratios)
                                exponential_penalty = tf.exp(ratio_deficit * 5.0) - 1.0  # Exponential scaling
                                
                                # EXTREME penalty for severe suppression (< 0.1)
                                severe_suppression_threshold = 0.1
                                severe_suppression = tf.maximum(0.0, severe_suppression_threshold - actual_ratios)
                                severe_penalty_weight = self.config.get("severe_suppression_penalty", 0.15)
                                
                                # Exponential penalty for very severe suppression (< 0.05)
                                extreme_suppression = tf.maximum(0.0, 0.05 - actual_ratios)
                                
                                variance_loss = (tf.reduce_mean(exponential_penalty) + 
                                               severe_penalty_weight * tf.reduce_mean(tf.exp(severe_suppression * 10.0)) +  # Exponential severe penalty
                                               severe_penalty_weight * tf.reduce_mean(tf.exp(extreme_suppression * 20.0)))  # Exponential extreme penalty
                            else:
                                # Fallback to ratio-based penalty
                                ratio_penalty = tf.reduce_mean(tf.maximum(0.0, target_ratios - actual_ratios))
                                
                                # STRONG penalty for severe suppression (< 0.1)
                                severe_suppression_threshold = 0.1
                                severe_suppression = tf.reduce_mean(tf.maximum(0.0, severe_suppression_threshold - actual_ratios))
                                severe_penalty_weight = self.config.get("severe_suppression_penalty", 0.15)
                                
                                # Exponential penalty for very severe suppression (< 0.05)
                                extreme_suppression = tf.reduce_mean(tf.maximum(0.0, 0.05 - actual_ratios))
                                
                                variance_loss = (ratio_penalty + 
                                               severe_penalty_weight * severe_suppression * 20.0 +  # 20x penalty for severe
                                               severe_penalty_weight * extreme_suppression * 100.0)  # 100x penalty for extreme
                        
                        # IMPROVEMENT 2: Enhanced uncertainty calibration
                        calibration_loss = 0.0
                        if self.config.get("uncertainty_calibration_weight", 0) > 0:
                            errors = tf.abs(y_true - means)
                            
                            # Target correlation approach
                            target_corr = self.config.get("uncertainty_target_correlation", 0.15)
                            
                            # Compute correlation between scales and errors for each target
                            calibration_losses = []
                            for i in range(len(self.target_columns)):
                                error_i = errors[:, i]
                                scale_i = scales[:, i]
                                
                                # Pearson correlation
                                error_mean = tf.reduce_mean(error_i)
                                scale_mean = tf.reduce_mean(scale_i)
                                
                                error_centered = error_i - error_mean
                                scale_centered = scale_i - scale_mean
                                
                                numerator = tf.reduce_mean(error_centered * scale_centered)
                                error_std = tf.sqrt(tf.reduce_mean(tf.square(error_centered)) + 1e-8)
                                scale_std = tf.sqrt(tf.reduce_mean(tf.square(scale_centered)) + 1e-8)
                                
                                correlation = numerator / (error_std * scale_std + 1e-8)
                                
                                # Penalize deviation from target correlation
                                calibration_losses.append(tf.square(correlation - target_corr))
                            
                            calibration_loss = tf.reduce_mean(tf.stack(calibration_losses))
                        
                        # Soft L2 regularization (working well)
                        delta_l2_loss = 0.0
                        if strategy.get("delta_l2_weight", 0) > 0:
                            # L2 penalty on prediction magnitudes to discourage extreme values
                            delta_l2_loss = tf.reduce_mean(tf.square(means))
                        
                        # IMPROVEMENT 3: Adaptive uncertainty scaling
                        adaptive_uncertainty_loss = 0.0
                        if self.config.get("adaptive_uncertainty", False):
                            # Encourage higher uncertainty when predictions are further from baseline
                            # Note: We can't access baseline directly, so use prediction magnitude as proxy
                            pred_magnitude = tf.abs(means)
                            uncertainty_should_increase = pred_magnitude / (tf.reduce_mean(pred_magnitude, axis=0) + 1e-6)
                            uncertainty_ratio = scales / (tf.reduce_mean(scales, axis=0) + 1e-6)
                            
                            # Penalize when uncertainty doesn't scale with prediction magnitude
                            adaptive_uncertainty_loss = tf.reduce_mean(tf.square(uncertainty_should_increase - uncertainty_ratio))
                        
                        # IMPROVEMENT 4: Ensemble diversity encouragement (NEW)
                        diversity_loss = 0.0
                        if strategy.get("diversity_encouragement", 0) > 0:
                            # Encourage this model to make different predictions from a "baseline" model
                            # Use a simple heuristic: penalize predictions that are too close to the mean
                            mean_prediction = tf.reduce_mean(means, axis=0, keepdims=True)
                            prediction_deviation = tf.abs(means - mean_prediction)
                            
                            # Encourage larger deviations (more diversity)
                            diversity_loss = -tf.reduce_mean(prediction_deviation)  # Negative to encourage larger deviations
                        
                        total_loss = (tf.reduce_mean(nll) + 
                                     strategy["delta_l2_weight"] * delta_l2_loss +  # Soft L2 regularization
                                     self.config.get("variance_encouragement_weight", 0) * variance_loss +  # EXTREME variance encouragement
                                     self.config.get("uncertainty_calibration_weight", 0) * calibration_loss +  # Uncertainty calibration
                                     0.01 * adaptive_uncertainty_loss +  # Adaptive uncertainty
                                     strategy.get("diversity_encouragement", 0) * diversity_loss)  # Ensemble diversity
                        
                        return total_loss
                    else:
                        return tf.reduce_mean(tf.square(y_true - y_pred))
                
                return hybrid_loss
            
            # Custom metric for delta MAE
            def delta_mae(y_true, y_pred):
                if self.config["use_probabilistic"]:
                    means = y_pred[:, :len(self.target_columns)]
                    return tf.reduce_mean(tf.abs(y_true - means))
                else:
                    return tf.reduce_mean(tf.abs(y_true - y_pred))
            
            def r2_metric(y_true, y_pred):
                if self.config["use_probabilistic"]:
                    means = y_pred[:, :len(self.target_columns)]
                    ss_res = tf.reduce_sum(tf.square(y_true - means))
                    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
                    return 1 - ss_res / (ss_tot + 1e-8)
                else:
                    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
                    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
                    return 1 - ss_res / (ss_tot + 1e-8)
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.config["lr"]),
                loss=create_hybrid_loss_with_strategy(strategy),
                metrics=[delta_mae, r2_metric]
            )
            
            print(f"Model {model_idx + 1} summary:")
            if model_idx == 0:  # Only show summary for first model
                model.summary()
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_r2_metric', patience=self.config["patience"], restore_best_weights=True, mode='max'),
                ReduceLROnPlateau(monitor='val_r2_metric', patience=8, factor=0.5, min_lr=1e-6, mode='max')
            ]
            
            # Train
            print(f"Training model {model_idx + 1} with {strategy['name']} strategy...")
            history = model.fit(
                [X_train_scaled, baselines_train], y_train,
                validation_data=([X_val_scaled, baselines_val], y_val),
                epochs=self.config["epochs"],
                batch_size=self.config["batch_size"],
                callbacks=callbacks,
                verbose=1 if model_idx == 0 else 0  # Only show progress for first model
            )
            
            # Store model and history
            self.models.append(model)
            ensemble_histories.append(history)
            
            print(f"✅ Model {model_idx + 1} ({strategy['name']}) training completed!")
        
        # Save ensemble components
        print("\n💾 Saving ensemble components...")
        self._save_ensemble_components()
        
        print(f"\n🎉 Ensemble training completed! {self.ensemble_size} models trained.")
        return ensemble_histories
    
    def _prepare_training_data(self):
        """Prepare training data (extracted from original train method)"""
        
        # Load and prepare data
        X, baselines, y, df = self.prepare_data()
        
        print(f"Training data shape: X={X.shape}, baselines={baselines.shape}, y={y.shape}")
        
        # Create chronological train/validation split
        print("Creating chronological train/validation split...")
        split_idx = int(0.8 * len(X))
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        baselines_train, baselines_val = baselines[:split_idx], baselines[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Scale model features (keeping baselines in real units)
        print("Scaling model features (keeping baselines in real units)...")
        
        # Flatten for scaling (skip first 3 categorical features)
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_numeric = X_train_flat[:, 3:]  # Skip categorical features
        
        self.scaler_x = StandardScaler()
        X_train_numeric_scaled = self.scaler_x.fit_transform(X_train_numeric)
        
        # Reconstruct with scaled numeric features
        X_train_scaled = np.concatenate([X_train_flat[:, :3], X_train_numeric_scaled], axis=1)
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        
        # Scale validation data
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        X_val_numeric = X_val_flat[:, 3:]
        X_val_numeric_scaled = self.scaler_x.transform(X_val_numeric)
        X_val_scaled = np.concatenate([X_val_flat[:, :3], X_val_numeric_scaled], axis=1)
        X_val_scaled = X_val_scaled.reshape(X_val.shape)
        
        return X_train_scaled, baselines_train, y_train, X_val_scaled, baselines_val, y_val
    
    def _build_model(self):
        """Build model (renamed from build_model for ensemble)"""
        return self.build_model()
    
    def _build_model_with_strategy(self, strategy):
        """Build model with strategy-specific parameters for ensemble diversity"""
        
        print(f"🏗️ Building Hybrid Spike-Driver MoE with {strategy['name']} strategy...")
        
        # Input layers
        sequence_input = Input(shape=(self.config["seq_len"], len(self.feature_columns)), name="sequence_input")
        baseline_input = Input(shape=(len(self.baseline_features),), name="baseline_input")
        
        # Extract categorical features for embedding
        player_ids = Lambda(lambda x: tf.cast(x[:, :, 0], tf.int32), name="player_ids")(sequence_input)
        team_ids = Lambda(lambda x: tf.cast(x[:, :, 1], tf.int32), name="team_ids")(sequence_input)
        opponent_ids = Lambda(lambda x: tf.cast(x[:, :, 2], tf.int32), name="opponent_ids")(sequence_input)
        
        # Embeddings
        player_embed = Embedding(len(self.player_mapping), 16, name="player_embed")(player_ids)
        team_embed = Embedding(len(self.team_mapping), 8, name="team_embed")(team_ids)
        opponent_embed = Embedding(len(self.opponent_mapping), 8, name="opponent_embed")(opponent_ids)
        
        # Numeric features (skip first 3 categorical)
        numeric_features = Lambda(lambda x: x[:, :, 3:], name="numeric_features")(sequence_input)
        
        # Combine features
        combined = Concatenate(axis=-1, name="combined_features")([
            player_embed, team_embed, opponent_embed, numeric_features
        ])
        
        # Project to model dimension
        x = Dense(self.config["d_model"], name="input_projection")(combined)
        
        # Transformer layers with strategy-specific dropout
        for i in range(self.config["n_layers"]):
            # Multi-head attention
            attn = MultiHeadAttention(
                num_heads=self.config["n_heads"],
                key_dim=self.config["d_model"] // self.config["n_heads"],
                name=f"attention_{i}"
            )(x, x)
            
            x = Add(name=f"add_attn_{i}")([x, attn])
            x = LayerNormalization(name=f"norm_attn_{i}")(x)
            
            # Feed forward
            ff = Dense(self.config["d_model"] * 2, activation="relu", name=f"ff1_{i}")(x)
            ff = Dropout(strategy["dropout"], name=f"dropout_ff_{i}")(ff)  # Strategy-specific dropout
            ff = Dense(self.config["d_model"], name=f"ff2_{i}")(ff)
            
            x = Add(name=f"add_ff_{i}")([x, ff])
            x = LayerNormalization(name=f"norm_ff_{i}")(x)
        
        # Use full sequence for routing (not just last timestep)
        sequence_repr = GlobalAveragePooling1D(name="sequence_pooling")(x)
        
        # Condition router/experts on baseline (concat baseline into sequence_repr)
        sequence_repr_with_baseline = Concatenate(name="sequence_baseline_concat")([sequence_repr, baseline_input])
        
        # Per-target spike indicators with enhanced sensitivity
        spike_features_last = Lambda(lambda x: x[:, -1, -len(self.spike_features):], name="spike_features")(sequence_input)
        
        # Enhanced spike detection with multiple pathways
        spike_base = Dense(32, activation="relu", name="spike_detection_base")(spike_features_last)
        spike_base = Dropout(strategy.get("dropout", 0.1), name="spike_detection_dropout")(spike_base)
        
        # Per-target spike indicators with sensitivity boost
        spike_indicators = Dense(len(self.target_columns), activation="sigmoid", name="per_target_spike_indicators")(spike_base)
        
        # Apply context-aware spike detection
        sensitivity = self.config.get("spike_detection_sensitivity", 1.0)
        
        # Enhanced context-aware spike detection for better regime separation
        # Use baseline values and recent trends to inform spike likelihood
        baseline_pts = baseline_input[:, 0:1]  # PTS baseline
        baseline_trb = baseline_input[:, 1:2]  # TRB baseline  
        baseline_ast = baseline_input[:, 2:3]  # AST baseline
        
        # More nuanced baseline spike factors
        baseline_spike_factor = tf.concat([
            tf.sigmoid((baseline_pts - 18.0) / 8.0),   # PTS spike factor (adjusted thresholds)
            tf.sigmoid((baseline_trb - 7.0) / 4.0),    # TRB spike factor
            tf.sigmoid((baseline_ast - 5.0) / 3.0)     # AST spike factor
        ], axis=1)
        
        # Get recent performance trends from spike features
        recent_trends = spike_features_last[:, -3:]  # Last 3 trend features
        trend_factor = tf.reduce_mean(tf.abs(recent_trends), axis=1, keepdims=True)
        trend_boost = tf.sigmoid(trend_factor - 0.5) * 0.2  # 0-20% boost based on trends
        
        # Combine raw indicators with baseline context and trends
        context_multiplier = 0.6 + 0.3 * baseline_spike_factor + tf.tile(trend_boost, [1, 3])
        context_aware_indicators = spike_indicators * context_multiplier
        spike_indicators = Lambda(lambda x: tf.minimum(x * sensitivity, 1.0), name="enhanced_context_spike_detection")(context_aware_indicators)
        
        # Enhanced spike routing with temperature scaling and strategy-specific parameters
        total_experts = self.config["num_experts"] + self.config["num_spike_experts"]
        router_logits = Dense(total_experts, name="router")(sequence_repr_with_baseline)
        
        # Apply strategy-specific temperature scaling for sharper decisions
        temperature = strategy.get("router_temperature", self.config.get("router_temperature", 1.0))
        router_logits_scaled = router_logits / temperature
        
        # Apply spike routing bias for better spike expert utilization
        spike_expert_mask = tf.concat([
            tf.zeros([tf.shape(router_logits_scaled)[0], self.config["num_experts"]]),
            tf.ones([tf.shape(router_logits_scaled)[0], self.config["num_spike_experts"]])
        ], axis=1)
        avg_spike_score = tf.reduce_mean(spike_indicators, axis=1, keepdims=True)  # Average across PTS/TRB/AST
        
        # Apply intelligent, graduated routing strength for spike experts
        base_routing_strength = strategy.get("spike_routing_strength", self.config.get("spike_routing_strength", 1.0))
        
        # FIXED: Much more conservative multi-threshold routing
        low_spike_threshold = 0.7   # High threshold for weak spikes
        med_spike_threshold = 0.85  # Very high threshold for medium spikes  
        high_spike_threshold = 0.95 # Extremely high threshold for strong spikes
        
        # Very conservative routing strengths
        low_spike_mask = tf.cast((avg_spike_score > low_spike_threshold) & (avg_spike_score <= med_spike_threshold), tf.float32)
        med_spike_mask = tf.cast((avg_spike_score > med_spike_threshold) & (avg_spike_score <= high_spike_threshold), tf.float32)
        high_spike_mask = tf.cast(avg_spike_score > high_spike_threshold, tf.float32)
        
        # Much weaker routing strengths to prevent collapse
        routing_strength = (
            low_spike_mask * base_routing_strength * 0.1 +   # 10% strength for low confidence
            med_spike_mask * base_routing_strength * 0.3 +   # 30% strength for medium confidence
            high_spike_mask * base_routing_strength * 0.6    # 60% strength for high confidence
        )
        
        spike_routing_bias = avg_spike_score * spike_expert_mask * routing_strength
        
        # Adjust router probabilities with spike bias
        adjusted_router_logits = router_logits_scaled + spike_routing_bias
        router_probs = Softmax(name="router_probs")(adjusted_router_logits)
        
        # Add gradient flow fix: ensure all experts get some gradient flow
        if self.config.get("gradient_flow_fix", False):
            # Add small uniform distribution to prevent complete expert dropout
            uniform_dist = tf.ones_like(router_probs) / total_experts
            router_probs = 0.97 * router_probs + 0.03 * uniform_dist  # Minimal intervention for natural specialization
        expert_outputs = []
        for i in range(total_experts):
            expert_type = "spike" if i >= self.config["num_experts"] else "regular"
            # Enhanced spike expert specialization - larger capacity for spike experts
            expert_dim = self.config["spike_expert_capacity"] if expert_type == "spike" else self.config["expert_dim"]
            expert = Dense(expert_dim, activation="relu", name=f"expert_{expert_type}_{i}_1")(sequence_repr_with_baseline)
            expert = Dropout(strategy["dropout"], name=f"expert_{expert_type}_{i}_dropout")(expert)  # Strategy-specific dropout
            
            if self.config["use_probabilistic"]:
                expert_out = Dense(len(self.target_columns) * 4, name=f"expert_{expert_type}_{i}_out")(expert)
            else:
                expert_out = Dense(len(self.target_columns), name=f"expert_{expert_type}_{i}_out")(expert)
            
            expert_outputs.append(expert_out)
        
        # Combine expert outputs with utilization balancing
        expert_stack = tf.stack(expert_outputs, axis=1)
        router_probs_expanded = tf.expand_dims(router_probs, axis=-1)
        
        # Add expert utilization balancing for better gradient flow
        if self.config.get("expert_utilization_balancing", False):
            # Compute expert usage statistics
            expert_usage = tf.reduce_mean(router_probs, axis=0)
            # Apply gentle balancing to underused experts (configurable)
            min_usage = self.config.get("min_expert_usage", 0.0)
            underused_boost_weight = self.config.get("underused_expert_boost", 0.1)
            underused_boost = tf.maximum(0.0, min_usage - expert_usage) * underused_boost_weight
            
            # Apply boost to router probabilities
            router_probs_balanced = router_probs + tf.expand_dims(underused_boost, 0)
            router_probs_balanced = router_probs_balanced / tf.reduce_sum(router_probs_balanced, axis=1, keepdims=True)
            
            router_probs_expanded = tf.expand_dims(router_probs_balanced, axis=-1)
        
        delta_output = tf.reduce_sum(expert_stack * router_probs_expanded, axis=1)
        
        # Apply hybrid output layer
        final_output = ConditionalSpikeOutput(
            use_probabilistic=self.config["use_probabilistic"],
            min_scale=self.config["min_scale"],
            max_scale_pts=self.config["max_scale_pts"],
            max_scale_trb=self.config["max_scale_trb"],
            max_scale_ast=self.config["max_scale_ast"],
            name="conditional_spike_output"
        )([delta_output, baseline_input, spike_indicators])
        
        # Create model
        model = Model(inputs=[sequence_input, baseline_input], outputs=final_output, name="HybridSpikeMoE")
        
        # Add enhanced MoE regularization losses to the model with strategy-specific parameters
        # Load balance loss: encourage equal expert usage (REDUCED weight for specialization)
        router_probs_mean = tf.reduce_mean(router_probs, axis=0)
        load_balance_loss_val = tf.reduce_sum(tf.square(router_probs_mean - 1.0/total_experts))
        model.add_loss(self.config["load_balance_weight"] * load_balance_loss_val)
        
        # Expert diversity penalty: discourage expert collapse and over-concentration
        if self.config.get("expert_diversity_weight", 0) > 0:
            min_usage = self.config.get("min_expert_usage", 0.0)
            max_usage = self.config.get("max_expert_concentration", 1.0)
            underuse_penalty = tf.nn.relu(min_usage - router_probs_mean)
            overuse_penalty = tf.nn.relu(router_probs_mean - max_usage)
            diversity_penalty = tf.reduce_mean(tf.square(underuse_penalty) + tf.square(overuse_penalty))
            model.add_loss(self.config["expert_diversity_weight"] * diversity_penalty)
            model.add_metric(diversity_penalty, name='expert_diversity_penalty')
        
        # FIXED: Encourage router entropy to prevent collapse
        router_entropy_val = -tf.reduce_mean(tf.reduce_sum(router_probs * tf.math.log(router_probs + 1e-8), axis=1))
        entropy_target = strategy.get("entropy_target", self.config.get("entropy_target", 1.5))
        
        # Penalize low entropy (collapsed routing) more heavily
        entropy_penalty = tf.maximum(0.0, entropy_target - router_entropy_val)
        entropy_loss = tf.square(entropy_penalty) * 2.0  # Extra penalty for collapse
        model.add_loss(self.config["router_z_loss_weight"] * entropy_loss)
        
        # Confidence targeting: encourage high-confidence decisions
        confidence_threshold = strategy.get("confidence_threshold", self.config.get("confidence_threshold", 0.5))
        max_probs = tf.reduce_max(router_probs, axis=1)
        confidence_loss = tf.reduce_mean(tf.maximum(0.0, confidence_threshold - max_probs))
        model.add_loss(0.01 * confidence_loss)
        
        # Gate keepalive: prevent experts from being completely unused (REDUCED weight)
        gate_keepalive_val = tf.reduce_mean(tf.reduce_sum(tf.cast(router_probs > 0.01, tf.float32), axis=1))
        model.add_loss(self.config["gate_keepalive_weight"] * (-gate_keepalive_val))
        
        # Spike frequency targeting: ensure adequate spike expert usage
        spike_expert_usage = tf.reduce_mean(tf.reduce_sum(router_probs[:, self.config["num_experts"]:], axis=1))
        spike_frequency_target = self.config.get("spike_frequency_target", 0.15)
        spike_frequency_loss = tf.square(spike_expert_usage - spike_frequency_target)
        model.add_loss(self.config.get("spike_frequency_weight", 0.01) * spike_frequency_loss)
        
        # Regime separation loss: encourage different routing for different spike levels
        if self.config.get("regime_separation_weight", 0) > 0:
            # High spike samples should route more to spike experts
            high_spike_samples = tf.cast(avg_spike_score > 0.6, tf.float32)
            low_spike_samples = tf.cast(avg_spike_score < 0.3, tf.float32)
            
            high_spike_routing = tf.reduce_sum(router_probs[:, self.config["num_experts"]:], axis=1)
            low_spike_routing = tf.reduce_sum(router_probs[:, :self.config["num_experts"]], axis=1)
            
            # Encourage high spike samples to use spike experts more
            high_spike_separation = tf.reduce_mean(high_spike_samples * (0.7 - high_spike_routing))
            # Encourage low spike samples to use regular experts more  
            low_spike_separation = tf.reduce_mean(low_spike_samples * (0.7 - low_spike_routing))
            
            regime_separation_loss = tf.maximum(0.0, high_spike_separation) + tf.maximum(0.0, low_spike_separation)
            model.add_loss(self.config["regime_separation_weight"] * regime_separation_loss)
            
            model.add_metric(regime_separation_loss, name='regime_separation_loss')
        
        # Spike routing prior loss: encourage spike experts when spike indicators are high
        avg_spike_score = tf.reduce_mean(spike_indicators, axis=1, keepdims=True)  # Average across PTS/TRB/AST
        spike_expert_mask = tf.concat([
            tf.zeros([tf.shape(router_probs)[0], self.config["num_experts"]]),
            tf.ones([tf.shape(router_probs)[0], self.config["num_spike_experts"]])
        ], axis=1)
        # FIXED: Penalize excessive spike routing (prevent collapse)
        spike_expert_usage = tf.reduce_sum(router_probs * spike_expert_mask, axis=1)
        target_spike_usage = 0.15  # Target 15% spike routing
        spike_routing_loss_val = tf.reduce_mean(tf.square(spike_expert_usage - target_spike_usage))
        model.add_loss(self.config["spike_routing_bias"] * spike_routing_loss_val)
        
        # Expert specialization loss: encourage experts to specialize on different patterns
        if self.config.get("expert_specialization_weight", 0) > 0:
            # Encourage different experts to activate for different input patterns
            expert_activations = tf.reduce_mean(router_probs, axis=0)  # Average activation per expert
            specialization_loss = -tf.reduce_sum(expert_activations * tf.math.log(expert_activations + 1e-8))
            model.add_loss(self.config["expert_specialization_weight"] * (-specialization_loss))
        
        # Add metric tracking for these losses
        model.add_metric(load_balance_loss_val, name='load_balance_loss')
        model.add_metric(router_entropy_val, name='router_entropy') 
        model.add_metric(confidence_loss, name='confidence_loss')
        model.add_metric(spike_frequency_loss, name='spike_frequency_loss')
        model.add_metric(gate_keepalive_val, name='gate_keepalive')
        model.add_metric(spike_routing_loss_val, name='spike_routing_loss')
        model.add_metric(max_probs, name='max_routing_confidence')
        model.add_metric(spike_expert_usage, name='spike_expert_usage')
                # Add diagnostic metrics for debugging router collapse
        model.add_metric(tf.reduce_mean(tf.reduce_max(router_probs, axis=1)), name='avg_max_prob')
        model.add_metric(tf.reduce_mean(tf.cast(tf.argmax(router_probs, axis=1) >= self.config["num_experts"], tf.float32)), name='spike_routing_rate')
        model.add_metric(tf.reduce_mean(avg_spike_score), name='avg_spike_score')
        
        # Add expert usage distribution
        for i in range(total_experts):
            model.add_metric(tf.reduce_mean(router_probs[:, i]), name=f'expert_{i}_usage')
        
        return model
    
    def _save_ensemble_components(self):
        """Save ensemble models and metadata with strategy info"""
        
        # Save each model in the ensemble
        for i, model in enumerate(self.models):
            model.save_weights(f"model/hybrid_spike_ensemble_{i}_weights.h5")
            print(f"✅ Model {i+1} weights saved: model/hybrid_spike_ensemble_{i}_weights.h5")
        
        # Save scaler
        joblib.dump(self.scaler_x, "model/hybrid_spike_scaler_x.pkl")
        print("✅ Scaler saved: model/hybrid_spike_scaler_x.pkl")
        
        # Save metadata with ensemble info and new fixes
        metadata = {
            "feature_columns": self.feature_columns,
            "baseline_features": self.baseline_features,
            "target_columns": self.target_columns,
            "config": self.config,
            "player_mapping": {str(k): int(v) for k, v in self.player_mapping.items()},
            "team_mapping": {str(k): int(v) for k, v in self.team_mapping.items()},
            "opponent_mapping": {str(k): int(v) for k, v in self.opponent_mapping.items()},
            "num_players": len(self.player_mapping),
            "num_teams": len(self.team_mapping),
            "num_opponents": len(self.opponent_mapping),
            "model_type": "hybrid_spike_driver_moe_ensemble_v2",
            "ensemble_size": self.ensemble_size,
            "use_probabilistic": self.config["use_probabilistic"],
            "fixes_applied": {
                "delta_centering": False,  # REMOVED: Was causing extreme negative predictions
                "delta_l2_regularization": True,  # ADDED: Soft L2 penalty instead
                "prediction_bounds": True,  # ADDED: Prevent negative/unrealistic values
                "student_t_likelihood": True,
                "ensemble_diversity": True,
                # NEW CRITICAL ROUTING FIXES
                "temperature_scaling": True,  # Sharp routing decisions via temperature
                "confidence_targeting": True,  # Target specific confidence levels
                "entropy_targeting": True,  # Target specific entropy levels (not just minimize)
                "spike_frequency_matching": True,  # Match target spike expert usage frequency
                "gradient_flow_fix": True,  # Ensure all experts get gradients
                "expert_specialization": True,  # Encourage expert specialization
                "reduced_expert_count": True,  # Fewer regular experts, more spike experts
                # ENHANCED IMPROVEMENTS
                "advanced_variance_encouragement": True,  # Ratio-based variance penalties
                "enhanced_spike_detection": True,  # Better spike sensitivity and routing
                "uncertainty_calibration": True,  # Target correlation-based calibration
                "adaptive_uncertainty": True,  # Uncertainty scales with prediction confidence
                "spike_context_features": True,  # Additional context for spike detection
            },
            "ensemble_strategies": [
                "high_confidence_spike_focused", "balanced_confidence_variance", "stable_confident_routing"
            ],
            "routing_improvements": {
                "entropy_reduction": "Target 0.6-0.9 vs previous 0.945",
                "confidence_increase": "Target 0.55-0.7 vs previous 0.471", 
                "spike_frequency": "Target 15% vs previous 8.5%",
                "expert_count": "6 regular + 4 spike vs previous 8 regular + 2 spike",
                "temperature_scaling": "0.3-0.7 for sharper decisions",
                "gradient_flow": "5% uniform distribution prevents expert dropout"
            },
            "ensemble_strategies": [
                "high_confidence_spike_focused", "balanced_confidence_variance", "stable_confident_routing"
            ],
            "routing_improvements": {
                "entropy_reduction": "Target 0.6-0.9 vs previous 0.945",
                "confidence_increase": "Target 0.55-0.7 vs previous 0.471", 
                "spike_frequency": "Target 15% vs previous 8.5%",
                "expert_count": "6 regular + 4 spike vs previous 8 regular + 2 spike",
                "temperature_scaling": "0.3-0.7 for sharper decisions",
                "gradient_flow": "5% uniform distribution prevents expert dropout"
            }
        }
        
        with open("model/hybrid_spike_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print("✅ Metadata saved: model/hybrid_spike_metadata.json")
        
        print(f"🎉 Ensemble of {self.ensemble_size} models saved successfully!")
        print("🔧 Applied fixes: L2 regularization, Prediction bounds, Student-t likelihood, Ensemble diversity")

def main():
    """Train ensemble hybrid spike-driver MoE model"""
    
    # Create trainer with ensemble
    trainer = HybridSpikeMoETrainer(ensemble_size=3)  # 3-model ensemble
    
    # Train ensemble
    histories = trainer.train()
    
    print("🎉 Ensemble training completed!")

if __name__ == "__main__":
    main()
    split_idx = int(0.8 * n_sequences)
            
    for i, (x, baseline, target) in enumerate(sequences):
        if i < split_idx:
            X_train.append(x)
            baselines_train.append(baseline)
            y_train.append(target)
        else:
            X_val.append(x)
            baselines_val.append(baseline)
            y_val.append(target)
        
        X_train = np.array(X_train, dtype=np.float32)
        X_val = np.array(X_val, dtype=np.float32)
        baselines_train = np.array(baselines_train, dtype=np.float32)
        baselines_val = np.array(baselines_val, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Scale only the model features (not baselines!)
        print("Scaling model features (keeping baselines in real units)...")
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        
        # Scale only non-categorical features
        self.scaler_x = StandardScaler()
        X_train_numeric = X_train_flat[:, 3:]  # Skip first 3 categorical
        X_val_numeric = X_val_flat[:, 3:]
        
        X_train_numeric_scaled = self.scaler_x.fit_transform(X_train_numeric)
        X_val_numeric_scaled = self.scaler_x.transform(X_val_numeric)
        
        # Reconstruct with categorical features
        X_train_scaled = np.concatenate([X_train_flat[:, :3], X_train_numeric_scaled], axis=1)
        X_val_scaled = np.concatenate([X_val_flat[:, :3], X_val_numeric_scaled], axis=1)
        
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_val_scaled = X_val_scaled.reshape(X_val.shape)
        
        # Build model
        self.model = self.build_model()
        
        # Create loss function with access to model components
        def create_hybrid_loss():
            def hybrid_loss(y_true, y_pred):
                if self.config["use_probabilistic"]:
                    # ConditionalSpikeOutput returns 15 dims:
                    # [normal_means(3), normal_scales(3), spike_means(3), spike_scales(3), spike_indicators(3)]
                    # We only use the first 6 (normal component) for loss calculation
                    means = y_pred[:, :len(self.target_columns)]
                    scales = y_pred[:, len(self.target_columns):2*len(self.target_columns)]
                    
                    # NLL loss
                    nll = 0.5 * tf.math.log(2 * np.pi * tf.square(scales))
                    nll += 0.5 * tf.square((y_true - means) / scales)
                    
                    # PROPER tail weighting (>1 for spikes)
                    spike_masks = []
                    for i, (stat, threshold) in enumerate(self.config["spike_thresholds"].items()):
                        mask = tf.cast(y_true[:, i] >= threshold, tf.float32)
                        weight = self.config["spike_loss_weights"][stat]  # Now >1
                        spike_masks.append(mask * weight + (1 - mask))
                    
                    spike_weights = tf.stack(spike_masks, axis=1)
                    nll = nll * spike_weights
                    
                    # Fix #3: Conditional variance encouragement (delta variance, not global)
                    # Note: We can't access baseline_input directly in loss, so use global variance with better targets
                    pred_vars = tf.math.reduce_variance(means, axis=0)
                    target_vars = tf.math.reduce_variance(y_true, axis=0)
                    
                    # Fix #2: More aggressive variance encouragement
                    target_ratios = tf.constant(self.config["target_variance_ratios"], dtype=tf.float32)  # [0.5, 0.6, 0.4]
                    actual_ratios = pred_vars / (target_vars + 1e-6)
                    delta_var_floor = tf.reduce_mean(tf.maximum(0.0, target_ratios - actual_ratios))
                    
                    # Additional variance penalty for severe suppression
                    variance_penalty = tf.reduce_mean(tf.maximum(0.0, 0.2 - actual_ratios))  # Penalize ratios < 0.2
                    
                    # UNCERTAINTY CALIBRATION: Encourage σ to correlate with errors
                    errors = tf.abs(y_true - means)
                    # Penalize when σ is low but error is high (anti-calibration)
                    calibration_penalty = tf.reduce_mean(tf.maximum(0.0, errors - scales))
                    
                    total_loss = (tf.reduce_mean(nll) + 
                                 self.config["variance_floor_weight"] * delta_var_floor +  # Conditional variance
                                 self.config["variance_penalty_weight"] * variance_penalty +  # Additional variance penalty
                                 0.1 * calibration_penalty)  # Uncertainty calibration
                    
                    return total_loss
                else:
                    return tf.reduce_mean(tf.square(y_true - y_pred))
            
            return hybrid_loss
        
        # Custom metric for delta MAE
        def delta_mae(y_true, y_pred):
            if self.config["use_probabilistic"]:
                # Extract means from first 3 dimensions
                means = y_pred[:, :len(self.target_columns)]
                return tf.reduce_mean(tf.abs(y_true - means))
            else:
                return tf.reduce_mean(tf.abs(y_true - y_pred))
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config["lr"]),
            loss=create_hybrid_loss(),
            metrics=[delta_mae]
        )
        
        print("Model summary:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=self.config["patience"], restore_best_weights=True),
            ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6)
        ]
        
        # Train
        print("Training hybrid model...")
        history = self.model.fit(
            [X_train_scaled, baselines_train], y_train,
            validation_data=([X_val_scaled, baselines_val], y_val),
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save components
        print("Saving model components...")
        
        # Save model weights instead of full model to avoid serialization issues
        self.model.save_weights("model/hybrid_spike_moe_weights.h5")
        
        # Save scaler
        joblib.dump(self.scaler_x, "model/hybrid_spike_scaler_x.pkl")
        
        # Save metadata
        metadata = {
            "feature_columns": self.feature_columns,
            "baseline_features": self.baseline_features,
            "target_columns": self.target_columns,
            "config": self.config,
            "player_mapping": {str(k): int(v) for k, v in self.player_mapping.items()},
            "team_mapping": {str(k): int(v) for k, v in self.team_mapping.items()},
            "opponent_mapping": {str(k): int(v) for k, v in self.opponent_mapping.items()},
            "num_players": len(self.player_mapping),
            "num_teams": len(self.team_mapping),
            "num_opponents": len(self.opponent_mapping),
            "model_type": "hybrid_spike_driver_moe",
            "use_probabilistic": self.config["use_probabilistic"]
        }
        
        with open("model/hybrid_spike_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Hybrid training completed!")
        print(f"✅ Model weights saved: model/hybrid_spike_moe_weights.h5")
        print(f"✅ Scaler saved: model/hybrid_spike_scaler_x.pkl")
        print(f"✅ Metadata saved: model/hybrid_spike_metadata.json")
