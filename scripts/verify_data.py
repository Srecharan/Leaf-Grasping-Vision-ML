import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def verify_training_data():
    data_path = os.path.expanduser('~/leaf_grasp_output/ml_training_data/training_data.pt')
    data = torch.load(data_path)
    
    print("=== Data Verification Report ===")
    
    # Check value ranges
    print("\nValue Ranges:")
    print(f"Depth range: [{data['depth_patches'].min():.3f}, {data['depth_patches'].max():.3f}]")
    print(f"Mask values: {torch.unique(data['mask_patches']).tolist()}")
    print(f"Score ranges:")
    for i in range(data['score_patches'].size(1)):
        print(f"  Channel {i}: [{data['score_patches'][:,i].min():.3f}, {data['score_patches'][:,i].max():.3f}]")
    
    # Check for NaN/Inf values
    print("\nNaN/Inf Check:")
    for key, tensor in data.items():
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        print(f"{key}: NaN: {has_nan}, Inf: {has_inf}")
    
    # Check patch center values
    print("\nPatch Center Analysis:")
    center = 16  # 32/2
    for i in range(len(data['depth_patches'])):
        depth_center = data['depth_patches'][i, center, center].item()
        mask_center = data['mask_patches'][i, center, center].item()
        print(f"Sample {i}: Depth={depth_center:.3f}, Mask={mask_center}")
        
    # Check grasp points are within bounds
    print("\nGrasp Point Bounds:")
    min_coords = torch.min(data['grasp_points'], dim=0)[0]
    max_coords = torch.max(data['grasp_points'], dim=0)[0]
    print(f"X range: [{min_coords[0]}, {max_coords[0]}]")
    print(f"Y range: [{min_coords[1]}, {max_coords[1]}]")
    
    # Label distribution
    print("\nLabel Distribution:")
    labels, counts = torch.unique(data['labels'], return_counts=True)
    for label, count in zip(labels, counts):
        print(f"Label {label}: {count} samples")
        
    return True

if __name__ == '__main__':
    verify_training_data()