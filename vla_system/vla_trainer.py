import torch
import torch.nn as nn
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import List, Dict, Tuple
import json
import os
import rospy
from PIL import Image
import cv2

class VLATrainer:
    def __init__(self, model_id="llava-hf/llava-v1.6-mistral-7b-hf", 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None
        self.peft_config = None
        self.setup_model()
        
    def setup_model(self):
        try:
            self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto" if self.device == 'cuda' else None,
                low_cpu_mem_usage=True
            )
            
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                               "gate_proj", "up_proj", "down_proj"]
            )
            
            self.model = get_peft_model(self.model, self.peft_config)
            rospy.loginfo(f"VLA trainer initialized on {self.device}")
            
        except Exception as e:
            rospy.logerr(f"Failed to setup VLA trainer: {e}")
            
    def create_training_data(self, candidates_data: List[Dict]) -> List[Dict]:
        training_samples = []
        
        for data in candidates_data:
            if 'image' not in data or 'candidates' not in data:
                continue
                
            image = data['image']
            candidates = data['candidates']
            ground_truth = data.get('ground_truth_ranking', list(range(len(candidates))))
            
            for i, candidate in enumerate(candidates):
                sample = {
                    'image': image,
                    'instruction': "Evaluate this leaf grasping candidate",
                    'candidate': candidate,
                    'score': self._calculate_target_score(i, ground_truth),
                    'prompt': self._create_training_prompt(candidate)
                }
                training_samples.append(sample)
                
        return training_samples
    
    def _calculate_target_score(self, candidate_idx: int, ranking: List[int]) -> float:
        if candidate_idx not in ranking:
            return 0.5
            
        position = ranking.index(candidate_idx)
        normalized_score = 1.0 - (position / len(ranking))
        return normalized_score
    
    def _create_training_prompt(self, candidate: Dict) -> str:
        prompt = f"""<|im_start|>system
You are an expert robotic vision system for leaf grasping.
<|im_end|>
<|im_start|>user
<image>
Evaluate this leaf grasp candidate:
Position: ({candidate.get('x', 0)}, {candidate.get('y', 0)})
Geometric score: {candidate.get('geometric_score', 0.5):.3f}
Clutter score: {candidate.get('clutter_score', 0.5):.3f}

Rate from 0.0 to 1.0 for grasping quality.
<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def prepare_training_batch(self, samples: List[Dict], batch_size: int = 4):
        batches = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            batches.append(self._process_batch(batch))
        return batches
    
    def _process_batch(self, batch: List[Dict]) -> Dict:
        images = []
        prompts = []
        targets = []
        
        for sample in batch:
            if isinstance(sample['image'], np.ndarray):
                image = Image.fromarray(cv2.cvtColor(sample['image'], cv2.COLOR_BGR2RGB))
            else:
                image = sample['image']
                
            images.append(image)
            prompts.append(sample['prompt'])
            targets.append(str(sample['score']))
            
        return {
            'images': images,
            'prompts': prompts,
            'targets': targets
        }
    
    def fine_tune(self, training_data: List[Dict], epochs: int = 3, 
                  learning_rate: float = 5e-5, save_path: str = None):
        
        if not self.model:
            rospy.logerr("Model not initialized")
            return
            
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        training_samples = self.create_training_data(training_data)
        batches = self.prepare_training_batch(training_samples)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(batches):
                
                for i in range(len(batch['images'])):
                    try:
                        inputs = self.processor(
                            batch['prompts'][i], 
                            batch['images'][i], 
                            return_tensors="pt"
                        ).to(self.device)
                        
                        labels = self.processor.tokenizer(
                            batch['targets'][i],
                            return_tensors="pt",
                            add_special_tokens=False
                        )['input_ids'].to(self.device)
                        
                        outputs = self.model(**inputs, labels=labels)
                        loss = outputs.loss
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                        
                    except Exception as e:
                        rospy.logwarn(f"Training step failed: {e}")
                        continue
                        
            avg_loss = total_loss / len(batches)
            rospy.loginfo(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
        if save_path:
            self.save_model(save_path)
            
    def save_model(self, save_path: str):
        try:
            os.makedirs(save_path, exist_ok=True)
            self.model.save_pretrained(save_path)
            rospy.loginfo(f"Model saved to {save_path}")
        except Exception as e:
            rospy.logerr(f"Failed to save model: {e}")
            
    def load_model(self, model_path: str):
        try:
            self.model = get_peft_model(self.model, self.peft_config)
            self.model.load_adapter(model_path)
            rospy.loginfo(f"Model loaded from {model_path}")
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            
    def generate_synthetic_data(self, num_samples: int = 50) -> List[Dict]:
        synthetic_data = []
        
        for i in range(num_samples):
            candidates = []
            for j in range(5):
                candidate = {
                    'x': np.random.randint(100, 1340),
                    'y': np.random.randint(100, 980),
                    'geometric_score': np.random.uniform(0.3, 0.9),
                    'clutter_score': np.random.uniform(0.2, 0.8),
                    'distance_score': np.random.uniform(0.4, 0.9)
                }
                candidates.append(candidate)
            
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            sample = {
                'image': dummy_image,
                'candidates': candidates,
                'ground_truth_ranking': sorted(range(5), 
                                             key=lambda x: candidates[x]['geometric_score'], 
                                             reverse=True)
            }
            synthetic_data.append(sample)
            
        return synthetic_data 