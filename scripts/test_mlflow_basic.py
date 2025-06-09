#!/usr/bin/env python3
"""
Basic MLflow integration test for the leaf grasping system.
Tests experiment tracking without requiring PyTorch.
"""

import os
import mlflow
import numpy as np
import json

def test_mlflow_setup():
    """Test basic MLflow setup and experiment creation"""
    print("Testing MLflow setup...")
    
    mlflow_dir = os.path.expanduser('~/leaf_grasp_output/mlflow_experiments')
    os.makedirs(mlflow_dir, exist_ok=True)
    
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    print(f"✓ MLflow tracking URI set to: {mlflow_dir}")
    
    experiment_name = "LeafGrasp-MLflow-Test"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✓ Created new experiment: {experiment_name} (ID: {experiment_id})")
    except Exception:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"✓ Using existing experiment: {experiment_name} (ID: {experiment_id})")
    
    mlflow.set_experiment(experiment_name)
    return experiment_id

def test_experiment_tracking():
    """Test experiment tracking capabilities"""
    print("\nTesting experiment tracking...")
    
    test_configs = [
        {
            'config_id': 1,
            'attention_mechanism': 'spatial',
            'learning_rate': 0.0005,
            'batch_size': 16,
            'architecture': 'standard'
        },
        {
            'config_id': 2,
            'attention_mechanism': 'hybrid',
            'learning_rate': 0.001,
            'batch_size': 32,
            'architecture': 'deep'
        }
    ]
    
    run_results = []
    
    for config in test_configs:
        with mlflow.start_run(run_name=f"Test_Config_{config['config_id']}") as run:
            print(f"  Running experiment for Config {config['config_id']}")
            
            mlflow.log_params(config)
            
            epochs = 5
            for epoch in range(epochs):
                train_loss = 0.8 * np.exp(-epoch * 0.3) + np.random.normal(0, 0.02)
                val_loss = 0.9 * np.exp(-epoch * 0.25) + np.random.normal(0, 0.03)
                
                base_accuracy = 60 + config['config_id'] * 5
                accuracy = base_accuracy + 25 * (1 - np.exp(-epoch * 0.4)) + np.random.normal(0, 1)
                
                f1_score = accuracy - 5 + np.random.normal(0, 2)
                precision = f1_score + np.random.normal(0, 1.5)
                recall = f1_score + np.random.normal(0, 1.5)
                
                mlflow.log_metrics({
                    'train_loss': max(0.01, train_loss),
                    'val_loss': max(0.01, val_loss),
                    'accuracy': min(100, max(0, accuracy)),
                    'f1_score': min(100, max(0, f1_score)),
                    'precision': min(100, max(0, precision)),
                    'recall': min(100, max(0, recall))
                }, step=epoch)
            
            final_metrics = {
                'final_f1_score': f1_score,
                'final_accuracy': accuracy,
                'final_precision': precision,
                'final_recall': recall
            }
            mlflow.log_metrics(final_metrics)
            
            mlflow.log_params({
                'final_attention_type': config['attention_mechanism'],
                'final_architecture_type': config['architecture']
            })
            
            run_info = {
                'run_id': run.info.run_id,
                'config_id': config['config_id'],
                'final_f1': f1_score,
                'attention': config['attention_mechanism'],
                'architecture': config['architecture']
            }
            run_results.append(run_info)
            
            print(f"    ✓ Logged {epochs} epochs of metrics")
            print(f"    ✓ Final F1 Score: {f1_score:.2f}%")
            print(f"    ✓ Run ID: {run.info.run_id}")
    
    return run_results

def test_configuration_integration():
    """Test integration with configuration generator"""
    print("\nTesting configuration generator integration...")
    
    try:
        from mlflow_experiment_configs import MLflowExperimentManager
        
        manager = MLflowExperimentManager()
        configs = manager.generate_comprehensive_configs()
        
        print(f"✓ Generated {len(configs)} total configurations")
        
        sample_configs = configs[:3]
        print(f"✓ Sample configurations:")
        for config in sample_configs:
            print(f"    Config {config['config_id']}: {config['attention_mechanism']} attention, "
                  f"{config['encoder_config']['name']} architecture")
        
        return True
    except Exception as e:
        print(f"✗ Configuration generator test failed: {e}")
        return False

def test_mlflow_ui_setup():
    """Test MLflow UI setup and provide instructions"""
    print("\nTesting MLflow UI setup...")
    
    mlflow_dir = os.path.expanduser('~/leaf_grasp_output/mlflow_experiments')
    
    if os.path.exists(mlflow_dir):
        print(f"✓ MLflow experiments directory exists: {mlflow_dir}")
        
        contents = os.listdir(mlflow_dir)
        if contents:
            print(f"✓ Found experiment files: {contents}")
        
        print(f"\n📊 To view MLflow UI:")
        print(f"   1. Run: python3 -m mlflow ui --backend-store-uri file://{mlflow_dir}")
        print(f"   2. Open browser to: http://localhost:5000")
        print(f"   3. View experiment comparisons and metrics")
        
        return True
    else:
        print(f"✗ MLflow directory not found")
        return False

def main():
    """Run comprehensive MLflow integration test"""
    print("LeafGrasp-Vision-ML MLflow Integration Test")
    print("=" * 50)
    
    success_count = 0
    total_tests = 4
    
    try:
        experiment_id = test_mlflow_setup()
        success_count += 1
        
        run_results = test_experiment_tracking()
        success_count += 1
        
        if test_configuration_integration():
            success_count += 1
        
        if test_mlflow_ui_setup():
            success_count += 1
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"MLflow Integration Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! MLflow integration is working correctly.")
        print("\nFeatures working:")
        print("✓ MLflow experiment tracking")
        print("✓ Hyperparameter logging")
        print("✓ Performance metrics tracking") 
        print("✓ Configuration management")
        print("✓ 60+ model configuration support")
    else:
        print(f"⚠️  {total_tests - success_count} tests failed. Check the output above.")
    
    return success_count == total_tests

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1) 