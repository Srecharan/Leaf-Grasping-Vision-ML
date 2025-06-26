import numpy as np
import torch
import rospy
from typing import List, Dict, Tuple, Optional
from .confidence_manager import ConfidenceManager

class HybridSelector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.confidence_manager = ConfidenceManager()
        
    def select_best_candidate(self, candidates: List[Dict], 
                            geometric_scores: List[float],
                            vla_scores: List[float],
                            vla_confidence: float) -> Dict:
        
        if not candidates:
            return None
            
        weights = self._calculate_weights(vla_confidence)
        hybrid_scores = self._compute_hybrid_scores(
            geometric_scores, vla_scores, weights
        )
        
        best_idx = np.argmax(hybrid_scores)
        best_candidate = candidates[best_idx].copy()
        best_candidate['hybrid_score'] = hybrid_scores[best_idx]
        best_candidate['vla_weight'] = weights['vla']
        best_candidate['geometric_weight'] = weights['geometric']
        
        rospy.loginfo(f"Selected candidate {best_idx} with hybrid score {hybrid_scores[best_idx]:.3f}")
        rospy.loginfo(f"Weights - VLA: {weights['vla']:.2f}, Geometric: {weights['geometric']:.2f}")
        
        return best_candidate
    
    def _calculate_weights(self, vla_confidence: float) -> Dict[str, float]:
        if vla_confidence > 0.8:
            vla_weight = 0.6
        elif vla_confidence > 0.5:
            vla_weight = 0.3
        elif vla_confidence > 0.2:
            vla_weight = 0.1
        else:
            vla_weight = 0.0
            
        geometric_weight = 1.0 - vla_weight
        
        return {
            'vla': vla_weight,
            'geometric': geometric_weight
        }
    
    def _compute_hybrid_scores(self, geometric_scores: List[float], 
                             vla_scores: List[float], 
                             weights: Dict[str, float]) -> List[float]:
        
        geometric_scores = np.array(geometric_scores)
        vla_scores = np.array(vla_scores)
        
        geometric_scores = self._normalize_scores(geometric_scores)
        vla_scores = self._normalize_scores(vla_scores)
        
        hybrid_scores = (weights['geometric'] * geometric_scores + 
                        weights['vla'] * vla_scores)
        
        return hybrid_scores.tolist()
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores
            
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score - min_score < 1e-6:
            return np.ones_like(scores) * 0.5
            
        return (scores - min_score) / (max_score - min_score)
    
    def get_selection_strategy(self, vla_confidence: float) -> str:
        if vla_confidence > 0.8:
            return "VLA_DOMINANT"
        elif vla_confidence > 0.5:
            return "BALANCED"
        elif vla_confidence > 0.2:
            return "GEOMETRIC_DOMINANT"
        else:
            return "GEOMETRIC_ONLY" 