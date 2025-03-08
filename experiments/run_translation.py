import asyncio
import os
from pathlib import Path
from typing import Dict, Optional

from translation_experiment import TranslationExperiment
from src.utils.logger import get_logger, VerboseLevel

logger = get_logger(__name__)

async def run_experiment(experiment_dir: str, model_id: str = "sonar-small"):
    """Run a translation experiment for a specific directory"""
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
            
        # Run experiments
        for exp_dir in experiment_dirs:
            logger.info(f"Running experiment in {exp_dir}")
            try:
                await run_experiment(exp_dir)
            except Exception as e:
                logger.error(f"Error in experiment {exp_dir}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())