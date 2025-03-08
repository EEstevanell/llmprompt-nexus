#!/usr/bin/env python
import os
import asyncio
from pathlib import Path
from typing import Optional, List
import argparse

from translation_experiment import TranslationExperiment
from src.utils.logger import get_logger
from src.models.registry import registry

logger = get_logger(__name__)

async def run_experiment(
    experiment_dir: str, 
    model_id: str = "sonar-small",
    batch_size: int = 10
) -> None:
    """Run a translation experiment for a specific directory"""
    experiment_path = Path(experiment_dir) / "experiment.yaml"
    
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment config not found at {experiment_path}")
    
    # Load API keys from environment
    api_keys = {
        "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY")
    }

    # Validate model exists
    if not registry.get_model(model_id):
        available_models = registry.list_models()
        raise ValueError(f"Model {model_id} not found. Available models: {available_models}")
    
    # Initialize and run experiment
    experiment = TranslationExperiment(
        experiment_path=experiment_path,
        model_id=model_id,
        batch_size=batch_size
    )
    
    await experiment.initialize(api_keys)
    await experiment.run()

def get_experiment_dirs() -> List[Path]:
    """Get all valid experiment directories"""
    experiments_root = Path(__file__).parent
    return [
        d for d in experiments_root.iterdir() 
        if d.is_dir() and (d / "experiment.yaml").exists()
    ]

async def main(args: argparse.Namespace) -> None:
    """Main entry point for running translation experiments"""
    try:
        experiment_dirs = get_experiment_dirs()
        if not experiment_dirs:
            logger.error("No valid experiment directories found")
            return

        if args.experiment:
            # Run specific experiment
            exp_dir = Path(__file__).parent / args.experiment
            if not exp_dir.exists() or not (exp_dir / "experiment.yaml").exists():
                logger.error(f"Invalid experiment directory: {args.experiment}")
                return
            experiment_dirs = [exp_dir]
            
        # Run experiments
        for exp_dir in experiment_dirs:
            logger.info(f"Running experiment in {exp_dir}")
            try:
                await run_experiment(
                    experiment_dir=exp_dir,
                    model_id=args.model,
                    batch_size=args.batch_size
                )
            except Exception as e:
                logger.error(f"Error in experiment {exp_dir}: {str(e)}")
                if not args.continue_on_error:
                    raise
                continue

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run translation experiments")
    parser.add_argument(
        "--experiment",
        help="Specific experiment directory to run (e.g., 'books' or 'euipo'). If not specified, all experiments will run.",
        type=str
    )
    parser.add_argument(
        "--model",
        help="Model ID to use for translations",
        default="sonar-small",
        type=str
    )
    parser.add_argument(
        "--batch-size",
        help="Number of texts to process in each batch",
        default=10,
        type=int
    )
    parser.add_argument(
        "--continue-on-error",
        help="Continue running other experiments if one fails",
        action="store_true"
    )
    
    args = parser.parse_args()
    asyncio.run(main(args))