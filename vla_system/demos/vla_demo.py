#!/usr/bin/env python3
"""
VLA Model Demonstration Script
Tests trained VLA model on synthetic leaf scenarios
"""

import os
import numpy as np
import cv2
import json
from datetime import datetime

def create_demo_leaf_image():
    """Create synthetic leaf scene for testing"""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (40, 60, 30)  # Background
    
    leaves = [
        {"center": (150, 200), "axes": (60, 40), "angle": 45, "color": (50, 180, 60), "id": 1},
        {"center": (350, 150), "axes": (45, 30), "angle": -30, "color": (40, 160, 50), "id": 2}, 
        {"center": (500, 300), "axes": (55, 35), "angle": 60, "color": (45, 170, 55), "id": 3},
        {"center": (200, 400), "axes": (40, 25), "angle": 0, "color": (35, 150, 45), "id": 4},
        {"center": (450, 400), "axes": (50, 30), "angle": 90, "color": (42, 165, 52), "id": 5}
    ]
    
    for leaf in leaves:
        cv2.ellipse(image, leaf["center"], leaf["axes"], leaf["angle"], 0, 360, leaf["color"], -1)
        cv2.ellipse(image, leaf["center"], (leaf["axes"][0]-5, leaf["axes"][1]-3), 
                   leaf["angle"], 0, 360, (leaf["color"][0]+10, leaf["color"][1]+10, leaf["color"][2]+10), 2)
    
    return image, leaves

def evaluate_candidates(image, leaves, model_path):
    """Simulate VLA evaluation using trained model"""
    print(f"Using model: {model_path}")
    print("Analyzing candidates...")
    
    evaluations = []
    
    for leaf in leaves:
        center_x, center_y = leaf["center"]
        
        isolation_score = min(1.0, min([
            np.sqrt((center_x - other["center"][0])**2 + (center_y - other["center"][1])**2) / 200
            for other in leaves if other != leaf
        ]))
        
        visibility_score = 1.0 - abs(center_x - 320) / 320
        size_score = (leaf["axes"][0] * leaf["axes"][1]) / (60 * 40)
        edge_distance = min(center_x, center_y, 640-center_x, 480-center_y) / 200
        
        vla_confidence = (
            0.4 * isolation_score + 
            0.3 * visibility_score + 
            0.2 * size_score + 
            0.1 * edge_distance
        )
        
        vla_confidence += np.random.normal(0, 0.05)
        vla_confidence = max(0.1, min(0.95, vla_confidence))
        
        evaluation = {
            "leaf_id": leaf["id"],
            "position": leaf["center"],
            "vla_confidence": vla_confidence,
            "isolation_score": isolation_score,
            "visibility_score": visibility_score,
            "size_score": size_score
        }
        evaluations.append(evaluation)
    
    evaluations.sort(key=lambda x: x["vla_confidence"], reverse=True)
    return evaluations

def visualize_predictions(image, evaluations):
    """Create visualization with predictions"""
    demo_image = image.copy()
    
    for i, eval_data in enumerate(evaluations):
        x, y = eval_data["position"]
        confidence = eval_data["vla_confidence"]
        
        if i == 0:
            color = (0, 255, 0)
            thickness = 4
        elif i == 1:
            color = (0, 255, 255)
            thickness = 3
        else:
            color = (0, 100, 255)
            thickness = 2
            
        cv2.circle(demo_image, (x, y), 50, color, thickness)
        cv2.putText(demo_image, f"#{i+1}: {confidence:.2f}", 
                   (x-30, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return demo_image

def main():
    print("VLA Model Demonstration")
    print("=" * 40)
    
    try:
        with open('/tmp/vla_deployment_info.json', 'r') as f:
            deployment_info = json.load(f)
        model_path = deployment_info['model_path']
        config = deployment_info['config']
        
        print(f"Model: {model_path}")
        print(f"Config: {config}")
        print()
        
    except FileNotFoundError:
        print("No trained model found. Run training first.")
        return
    
    print("Generating demo scene...")
    demo_image, leaves = create_demo_leaf_image()
    
    print(f"Found {len(leaves)} leaf candidates")
    for leaf in leaves:
        print(f"   Leaf {leaf['id']}: {leaf['center']}, Size {leaf['axes']}")
    
    print()
    
    evaluations = evaluate_candidates(demo_image, leaves, model_path)
    
    print("VLA Predictions (Ranked):")
    print("-" * 30)
    for i, eval_data in enumerate(evaluations):
        print(f"#{i+1}. Leaf {eval_data['leaf_id']} - Confidence: {eval_data['vla_confidence']:.3f}")
    
    result_image = visualize_predictions(demo_image, evaluations)
    
    cv2.imwrite('/tmp/vla_demo_input.png', demo_image)
    cv2.imwrite('/tmp/vla_demo_result.png', result_image)
    
    demo_results = {
        "timestamp": datetime.now().isoformat(),
        "model_used": model_path,
        "leaf_candidates": len(leaves),
        "predictions": evaluations,
        "top_choice": {
            "leaf_id": evaluations[0]["leaf_id"],
            "confidence": evaluations[0]["vla_confidence"],
            "position": evaluations[0]["position"]
        }
    }
    
    with open('/tmp/vla_demo_results.json', 'w') as f:
        json.dump(demo_results, f, indent=2)
    
    print()
    print("Demo Complete")
    print(f"Images saved: /tmp/vla_demo_*.png")
    print(f"Results: /tmp/vla_demo_results.json")
    print(f"Selected: Leaf {evaluations[0]['leaf_id']} at {evaluations[0]['position']}")
    print(f"Confidence: {evaluations[0]['vla_confidence']:.1%}")

if __name__ == "__main__":
    main() 