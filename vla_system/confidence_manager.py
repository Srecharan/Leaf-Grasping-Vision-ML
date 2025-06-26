import numpy as np
import torch
from typing import List, Dict
import rospy

class ConfidenceManager:
    def __init__(self):
        self.confidence_history = []
        self.max_history = 10
        
    def calculate_confidence(self, vla_scores: List[float], 
                           geometric_scores: List[float]) -> float:
        
        if not vla_scores or not geometric_scores:
            return 0.0
            
        score_consistency = self._calculate_consistency(vla_scores, geometric_scores)
        score_variance = self._calculate_variance(vla_scores)
        score_magnitude = self._calculate_magnitude(vla_scores)
        
        confidence = (0.4 * score_consistency + 
                     0.3 * (1 - score_variance) + 
                     0.3 * score_magnitude)
        
        confidence = np.clip(confidence, 0.0, 1.0)
        self._update_history(confidence)
        
        return confidence
    
    def _calculate_consistency(self, vla_scores: List[float], 
                             geometric_scores: List[float]) -> float:
        vla_array = np.array(vla_scores)
        geo_array = np.array(geometric_scores)
        
        if len(vla_array) < 2:
            return 0.5
            
        vla_norm = (vla_array - np.min(vla_array)) / (np.max(vla_array) - np.min(vla_array) + 1e-6)
        geo_norm = (geo_array - np.min(geo_array)) / (np.max(geo_array) - np.min(geo_array) + 1e-6)
        
        correlation = np.corrcoef(vla_norm, geo_norm)[0, 1]
        if np.isnan(correlation):
            return 0.5
            
        return (correlation + 1) / 2
    
    def _calculate_variance(self, scores: List[float]) -> float:
        if len(scores) < 2:
            return 1.0
            
        variance = np.var(scores)
        normalized_variance = variance / (np.mean(scores) + 1e-6)
        
        return np.clip(normalized_variance, 0.0, 1.0)
    
    def _calculate_magnitude(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
            
        max_score = np.max(scores)
        score_range = np.max(scores) - np.min(scores)
        
        magnitude = max_score * (1 + score_range / 2)
        return np.clip(magnitude, 0.0, 1.0)
    
    def _update_history(self, confidence: float):
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > self.max_history:
            self.confidence_history.pop(0)
    
    def get_running_confidence(self) -> float:
        if not self.confidence_history:
            return 0.0
            
        return np.mean(self.confidence_history[-5:])
    
    def is_stable(self, threshold: float = 0.1) -> bool:
        if len(self.confidence_history) < 3:
            return False
            
        recent = self.confidence_history[-3:]
        return np.std(recent) < threshold 