import os
import yaml
import csv
import pandas as pd
from pathlib import Path

def load_experiment(experiment_id, data_directory="."):
    """Load experiment by ID"""
    experiment_dir = os.path.join(data_directory, experiment_id)
    
    if not os.path.exists(experiment_dir):
        raise ValueError(f"Experiment {experiment_id} not found")
        
    # Load experiment.yaml
    with open(os.path.join(experiment_dir, "experiment.yaml"), 'r', encoding='utf-8') as f:
        experiment_config = yaml.safe_load(f)
        
    # Load data.csv
    data_path = os.path.join(experiment_dir, "data.csv")
    data_df = pd.read_csv(data_path)
    
    return {
        "config": experiment_config,
        "data": data_df
    }

def list_experiments(data_directory="."):
    """List all available experiments"""
    return [d for d in os.listdir(data_directory) 
            if os.path.isdir(os.path.join(data_directory, d)) and 
            os.path.exists(os.path.join(data_directory, d, "experiment.yaml"))]

def query_experiment(experiment_id, data_directory=".", filter_phrase=None):
    """Query experiment by ID with optional phrase filter"""
    experiment = load_experiment(experiment_id, data_directory)
    
    if filter_phrase:
        filtered_data = experiment["data"][experiment["data"]["source_phrase"].str.contains(filter_phrase, case=False)]
        experiment["data"] = filtered_data
        
    return experiment

async def load_experiment_config(experiment_path: Path) -> dict:
    """Load experiment configuration from yaml file"""
    with open(experiment_path, 'r') as f:
        return yaml.safe_load(f)

async def load_translation_data(data_path: Path) -> pd.DataFrame:
    """Load translation data from CSV file"""
    return pd.read_csv(data_path)
