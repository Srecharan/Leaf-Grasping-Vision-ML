#!/usr/bin/env python3
"""
MLflow experiment configuration generator for leaf grasping system.
Generates comprehensive hyperparameter configurations for systematic experimentation.
"""

import itertools
from typing import Dict, Any, List
import json
import os

class MLflowExperimentManager:
    """Manages comprehensive hyperparameter experiments for the leaf grasping system."""
    
    def __init__(self):
        self.experiment_name = "LeafGrasp-Vision-ML-Self-Supervised"
        self.base_config = self._get_base_configuration()
        
    def _get_base_configuration(self) -> Dict[str, Any]:
        """Base configuration with proven defaults"""
        return {
            'max_epochs': 150,
            'early_stopping_patience': 15,
            'scheduler_patience': 5,
            'min_delta': 0.001,
            'gradient_clip': 1.0,
            'train_val_split': 0.8,
            'seed': 42
        }
    
    def get_attention_mechanism_configs(self) -> List[Dict[str, Any]]:
        """Generate configurations for different attention mechanisms."""
        attention_configs = [
            {
                'type': 'spatial',
                'description': 'Spatial attention focusing on location-based features',
                'expected_benefit': 'Better grasp point localization'
            },
            {
                'type': 'channel', 
                'description': 'Channel attention focusing on feature importance',
                'expected_benefit': 'Better feature discrimination'
            },
            {
                'type': 'hybrid',
                'description': 'Combined spatial and channel attention',
                'expected_benefit': 'Best of both mechanisms'
            },
            {
                'type': 'none',
                'description': 'No attention mechanism (baseline)',
                'expected_benefit': 'Computational efficiency baseline'
            }
        ]
        return attention_configs
    
    def get_confidence_weighting_strategies(self) -> List[Dict[str, Any]]:
        """Generate confidence weighting strategies for hybrid CV-ML approach."""
        strategies = []
        confidence_weights = [0.1, 0.2, 0.3, 0.4]  # ML weight (CV gets 1-weight)
        
        for weight in confidence_weights:
            strategy = {
                'ml_weight': weight,
                'cv_weight': 1.0 - weight,
                'description': f'ML confidence: {weight*100:.0f}%, CV baseline: {(1-weight)*100:.0f}%',
                'use_case': self._get_weight_use_case(weight)
            }
            strategies.append(strategy)
        
        return strategies
    
    def _get_weight_use_case(self, ml_weight: float) -> str:
        """Determine use case based on ML weight"""
        if ml_weight <= 0.1:
            return "Conservative: Rely heavily on proven geometric algorithms"
        elif ml_weight <= 0.2:
            return "Balanced: Moderate ML influence with geometric safety"
        elif ml_weight <= 0.3:
            return "Progressive: Increased ML confidence with CV backup"
        else:
            return "Aggressive: High ML confidence for edge cases"
    
    def get_architecture_variations(self) -> List[Dict[str, Any]]:
        """Generate different CNN architecture configurations."""
        architectures = [
            {
                'name': 'lightweight',
                'filters': [32, 64, 128],
                'description': 'Lightweight architecture for real-time inference',
                'target_use': 'Edge deployment, fast inference',
                'expected_params': '~50K parameters'
            },
            {
                'name': 'standard',
                'filters': [64, 128, 256],
                'description': 'Standard architecture balancing capacity and speed',
                'target_use': 'Production deployment with good performance',
                'expected_params': '~200K parameters'
            },
            {
                'name': 'deep',
                'filters': [64, 128, 256, 512],
                'description': 'Deeper architecture for complex feature learning',
                'target_use': 'High-accuracy scenarios, complex leaves',
                'expected_params': '~500K parameters'
            },
            {
                'name': 'wide',
                'filters': [128, 256, 512],
                'description': 'Wide architecture for rich feature representation',
                'target_use': 'Research scenarios, maximum accuracy',
                'expected_params': '~800K parameters'
            }
        ]
        return architectures
    
    def generate_comprehensive_configs(self) -> List[Dict[str, Any]]:
        """Generate comprehensive configurations for systematic experimentation."""
        
        # Hyperparameter ranges
        learning_rates = [0.0001, 0.0005, 0.001, 0.002]
        batch_sizes = [8, 16, 32]
        weight_decays = [0.01, 0.001, 0.0001]
        pos_weights = [1.5, 2.0, 2.5, 3.0]  # Class imbalance handling
        
        # Get component configurations
        attention_configs = self.get_attention_mechanism_configs()
        confidence_strategies = self.get_confidence_weighting_strategies()
        architectures = self.get_architecture_variations()
        
        # Generate systematic combinations
        configs = []
        config_id = 1
        
        # Primary sweep: Attention mechanisms + Architectures
        for attention in attention_configs:
            for arch in architectures:
                for conf_strategy in confidence_strategies[:2]:  # Limit to 2 confidence strategies
                    for lr in learning_rates[:2]:  # Limit to 2 learning rates
                        for bs in batch_sizes[:2]:  # Limit to 2 batch sizes
                            config = {
                                **self.base_config,
                                'config_id': config_id,
                                'experiment_group': 'attention_architecture_sweep',
                                'learning_rate': lr,
                                'batch_size': bs,
                                'weight_decay': weight_decays[0],  # Use default
                                'pos_weight': pos_weights[1],      # Use default 2.0
                                'attention_mechanism': attention['type'],
                                'attention_description': attention['description'],
                                'confidence_weight': conf_strategy['ml_weight'],
                                'confidence_strategy': conf_strategy['description'],
                                'encoder_config': {
                                    'filters': arch['filters'],
                                    'name': arch['name']
                                },
                                'architecture_description': arch['description'],
                                'expected_benefit': attention['expected_benefit'],
                                'target_use_case': arch['target_use']
                            }
                            configs.append(config)
                            config_id += 1
        
        # Secondary sweep: Hyperparameter optimization for best architectures
        best_attention = ['spatial', 'hybrid']  # Best performing attention types
        best_arch = architectures[1:3]  # Standard and deep architectures
        
        for attention_type in best_attention:
            for arch in best_arch:
                for lr in learning_rates:
                    for wd in weight_decays:
                        for pw in pos_weights:
                            config = {
                                **self.base_config,
                                'config_id': config_id,
                                'experiment_group': 'hyperparameter_optimization',
                                'learning_rate': lr,
                                'batch_size': 16,  # Fixed optimal batch size
                                'weight_decay': wd,
                                'pos_weight': pw,
                                'attention_mechanism': attention_type,
                                'confidence_weight': 0.3,  # Fixed optimal confidence weight
                                'encoder_config': {
                                    'filters': arch['filters'],
                                    'name': arch['name']
                                },
                                'optimization_focus': f'LR={lr}, WD={wd}, PW={pw}'
                            }
                            configs.append(config)
                            config_id += 1
                            
                            if len(configs) >= 64:
                                break
                    if len(configs) >= 64:
                        break
                if len(configs) >= 64:
                    break
            if len(configs) >= 64:
                break
        
        return configs
    
    def generate_experiment_plan(self) -> Dict[str, Any]:
        """Generate comprehensive experiment plan documenting the systematic approach."""
        configs = self.generate_comprehensive_configs()
        
        # Analyze configuration distribution
        attention_distribution = {}
        architecture_distribution = {}
        experiment_groups = {}
        
        for config in configs:
            # Count attention mechanisms
            att_type = config['attention_mechanism']
            attention_distribution[att_type] = attention_distribution.get(att_type, 0) + 1
            
            # Count architectures
            arch_name = config['encoder_config']['name']
            architecture_distribution[arch_name] = architecture_distribution.get(arch_name, 0) + 1
            
            # Count experiment groups
            group = config['experiment_group']
            experiment_groups[group] = experiment_groups.get(group, 0) + 1
        
        experiment_plan = {
            'experiment_overview': {
                'name': self.experiment_name,
                'total_configurations': len(configs),
                'objective': 'Systematic optimization of attention-based CNN for leaf grasp point prediction',
                'methodology': 'Self-supervised learning with geometric algorithm as teacher'
            },
            'experimental_design': {
                'attention_mechanisms_tested': list(attention_distribution.keys()),
                'architecture_variations': list(architecture_distribution.keys()),
                'experiment_groups': list(experiment_groups.keys()),
                'hyperparameter_ranges': {
                    'learning_rate': [0.0001, 0.002],
                    'batch_size': [8, 32],
                    'weight_decay': [0.0001, 0.01],
                    'pos_weight': [1.5, 3.0],
                    'confidence_weight': [0.1, 0.4]
                }
            },
            'distribution_analysis': {
                'attention_distribution': attention_distribution,
                'architecture_distribution': architecture_distribution,
                'experiment_groups': experiment_groups
            },
            'research_contributions': {
                'self_supervised_framework': 'Traditional CV as expert teacher for ML training',
                'hybrid_decision_making': 'Dynamic confidence weighting between CV and ML',
                'systematic_optimization': 'Comprehensive configuration evaluation',
                'attention_mechanisms': 'Spatial, channel, and hybrid attention comparison',
                'production_readiness': 'Multiple architecture variants for different deployment scenarios'
            },
            'expected_outcomes': {
                'performance_metrics': ['F1 Score > 94%', 'Precision > 92%', 'Recall > 97%'],
                'optimal_configuration': 'Best performing attention mechanism and architecture',
                'confidence_strategy': 'Optimal CV-ML weighting for robust decisions',
                'deployment_recommendations': 'Architecture choices for different use cases'
            },
            'configurations': configs
        }
        
        return experiment_plan
    
    def save_experiment_plan(self, output_dir: str = None):
        """Save the complete experiment plan to files"""
        if output_dir is None:
            output_dir = os.path.expanduser('~/leaf_grasp_output/mlflow_experiments')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save experiment plan
        plan = self.generate_experiment_plan()
        
        # Save full plan
        plan_path = os.path.join(output_dir, 'experiment_plan.json')
        with open(plan_path, 'w') as f:
            json.dump(plan, f, indent=2)
        
        # Save configurations only
        configs_path = os.path.join(output_dir, 'configurations.json')
        with open(configs_path, 'w') as f:
            json.dump(plan['configurations'], f, indent=2)
        
        # Save summary
        summary_path = os.path.join(output_dir, 'experiment_summary.md')
        self._generate_markdown_summary(plan, summary_path)
        
        print(f"Experiment plan saved to:")
        print(f"  - Full plan: {plan_path}")
        print(f"  - Configurations: {configs_path}")
        print(f"  - Summary: {summary_path}")
        
        return plan_path, configs_path, summary_path
    
    def _generate_markdown_summary(self, plan: Dict[str, Any], output_path: str):
        """Generate a markdown summary of the experiment plan"""
        with open(output_path, 'w') as f:
            f.write("# LeafGrasp-Vision-ML MLflow Experiment Plan\n\n")
            
            f.write("## Overview\n")
            f.write(f"- **Experiment Name**: {plan['experiment_overview']['name']}\n")
            f.write(f"- **Total Configurations**: {plan['experiment_overview']['total_configurations']}\n")
            f.write(f"- **Objective**: {plan['experiment_overview']['objective']}\n")
            f.write(f"- **Methodology**: {plan['experiment_overview']['methodology']}\n\n")
            
            f.write("## Experimental Design\n")
            f.write("### Attention Mechanisms\n")
            for att_type, count in plan['distribution_analysis']['attention_distribution'].items():
                f.write(f"- **{att_type.title()}**: {count} configurations\n")
            
            f.write("\n### Architecture Variations\n")
            for arch_type, count in plan['distribution_analysis']['architecture_distribution'].items():
                f.write(f"- **{arch_type.title()}**: {count} configurations\n")
            
            f.write("\n### Experiment Groups\n")
            for group, count in plan['distribution_analysis']['experiment_groups'].items():
                f.write(f"- **{group.replace('_', ' ').title()}**: {count} configurations\n")
            
            f.write("\n## Research Contributions\n")
            for contribution, description in plan['research_contributions'].items():
                f.write(f"- **{contribution.replace('_', ' ').title()}**: {description}\n")
            
            f.write("\n## Expected Outcomes\n")
            f.write("### Performance Targets\n")
            for metric in plan['expected_outcomes']['performance_metrics']:
                f.write(f"- {metric}\n")
            
            f.write(f"\n### Optimization Goals\n")
            f.write(f"- **{plan['expected_outcomes']['optimal_configuration']}**\n")
            f.write(f"- **{plan['expected_outcomes']['confidence_strategy']}**\n")
            f.write(f"- **{plan['expected_outcomes']['deployment_recommendations']}**\n")
            
            f.write("\n## MLflow Tracking\n")
            f.write("Each configuration will be tracked with the following metrics:\n")
            f.write("- Training and validation loss curves\n")
            f.write("- Precision, recall, and F1 score\n")
            f.write("- Class-wise accuracy (positive/negative samples)\n")
            f.write("- Attention mechanism effectiveness\n")
            f.write("- Model artifacts and training plots\n")
            f.write("- Hyperparameter combinations and results\n")

def main():
    """Generate and save the comprehensive experiment plan"""
    manager = MLflowExperimentManager()
    
    print("Generating comprehensive MLflow experiment plan...")
    
    plan_path, configs_path, summary_path = manager.save_experiment_plan()
    
    plan = manager.generate_experiment_plan()
    print(f"\nExperiment Plan Summary:")
    print(f"- Total Configurations: {plan['experiment_overview']['total_configurations']}")
    print(f"- Attention Mechanisms: {list(plan['distribution_analysis']['attention_distribution'].keys())}")
    print(f"- Architecture Variants: {list(plan['distribution_analysis']['architecture_distribution'].keys())}")
    print(f"- Experiment Groups: {list(plan['distribution_analysis']['experiment_groups'].keys())}")
    
    print(f"\nNext Steps:")
    print(f"1. Run: python train_model_mlflow.py --full-optimization")
    print(f"2. Monitor progress: mlflow ui --backend-store-uri file://{os.path.expanduser('~/leaf_grasp_output/mlflow_experiments')}")
    print(f"3. Analyze results using the generated configuration files")

if __name__ == '__main__':
    main() 