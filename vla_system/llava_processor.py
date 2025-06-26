import torch
import torch.nn.functional as F
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import numpy as np
import rospy
import cv2
from typing import List, Dict, Tuple, Optional
import json

class LLaVAProcessor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.processor = None
        self.load_model()
        
    def load_model(self):
        try:
            model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
            self.processor = LlavaNextProcessor.from_pretrained(model_id)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            rospy.loginfo(f"LLaVA model loaded on {self.device}")
        except Exception as e:
            rospy.logwarn(f"Failed to load LLaVA model: {e}")
            self.model = None
            
    def evaluate_candidates(self, image: np.ndarray, candidates: List[Dict], 
                          instruction: str = "Select the best leaf for grasping") -> List[float]:
        if self.model is None:
            return [0.5] * len(candidates)
            
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            scores = []
            for candidate in candidates:
                prompt = self._create_evaluation_prompt(candidate, instruction)
                score = self._evaluate_single_candidate(pil_image, prompt)
                scores.append(score)
                
            return self._normalize_scores(scores)
            
        except Exception as e:
            rospy.logwarn(f"VLA evaluation failed: {e}")
            return [0.5] * len(candidates)
    
    def _create_evaluation_prompt(self, candidate: Dict, instruction: str) -> str:
        prompt = f"""<|im_start|>system
You are an expert robotic vision system evaluating leaf grasp candidates.
<|im_end|>
<|im_start|>user
<image>
Task: {instruction}

Candidate details:
- Position: ({candidate.get('x', 0)}, {candidate.get('y', 0)})
- Geometric score: {candidate.get('geometric_score', 0.5):.3f}
- Clutter score: {candidate.get('clutter_score', 0.5):.3f}
- Distance score: {candidate.get('distance_score', 0.5):.3f}

Rate this candidate from 0.0 to 1.0 for grasping suitability. Consider:
1. Leaf isolation and accessibility
2. Surface quality for stable grasping
3. Positioning relative to other leaves

Respond with only a decimal number between 0.0 and 1.0.
<|im_end|>
<|im_start|>assistant
"""
        return prompt
        
    def _evaluate_single_candidate(self, image: Image.Image, prompt: str) -> float:
        try:
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=0.1
                )
                
            response = self.processor.decode(output[0], skip_special_tokens=True)
            response = response.split("assistant")[-1].strip()
            
            try:
                score = float(response)
                return np.clip(score, 0.0, 1.0)
            except:
                return 0.5
                
        except Exception as e:
            rospy.logwarn(f"Single candidate evaluation failed: {e}")
            return 0.5
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
            
        scores = np.array(scores)
        if np.std(scores) < 1e-6:
            return [0.5] * len(scores)
            
        normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        return normalized.tolist()
    
    def get_confidence(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
            
        scores = np.array(scores)
        max_score = np.max(scores)
        score_range = np.max(scores) - np.min(scores)
        
        confidence = max_score * (1 + score_range)
        return np.clip(confidence, 0.0, 1.0) 