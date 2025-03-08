import asyncio
import os
from pathlib import Path
from typing import Dict, Optional, List
from translation_experiment import TranslationExperiment
from src.models.registry import registry
from src.utils.logger import get_logger, VerboseLevel

logger = get_logger(__name__)

async def run_experiment(experiment_dir: str, model_id: str):
    """Run a translation experiment for a specific directory with a specific model"""
    experiment_path = Path(experiment_dir) / "experiment.yaml"
    
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment config not found at {experiment_path}")
    
    # Load API keys from environment
    api_keys = {
        "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY")
    }
    
    # Initialize and run experiment
    experiment = TranslationExperiment(experiment_path, model_id)
    await experiment.initialize(api_keys)
    await experiment.run()

async def run_experiments_with_models(experiment_dirs: List[Path], model_ids: List[str]):
    """Run experiments for each directory with each model"""
    for exp_dir in experiment_dirs:
        logger.info(f"Running experiments in {exp_dir}")
        for model_id in model_ids:
            try:
                logger.info(f"Running with model: {model_id}")
                await run_experiment(exp_dir, model_id)
            except Exception as e:
                logger.error(f"Error in experiment {exp_dir} with model {model_id}: {str(e)}")
                continue

async def main():
    """Main entry point for running translation experiments"""
    try:
        # Get list of experiment directories
        experiments_root = Path("experiments")
        experiment_dirs = [d for d in experiments_root.iterdir() 
                         if d.is_dir() and (d / "experiment.yaml").exists()]
        
        if not experiment_dirs:
            logger.error("No valid experiment directories found")
            return
        
        # Define models to use for experiments
        model_ids = ["sonar", "sonar-pro"]
        
        # Validate models exist in registry
        valid_models = []
        for model_id in model_ids:
            try:
                model_config = registry.get_model(model_id)
                valid_models.append(model_id)
                logger.info(f"Validated model: {model_id}")
            except ValueError as e:
                logger.error(f"Model validation failed for {model_id}: {str(e)}")
        
        if not valid_models:
            logger.error("No valid models available for experiments")
            return
            
        # Run experiments with validated models
        await run_experiments_with_models(experiment_dirs, valid_models)
                
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())