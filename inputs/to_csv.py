import os
import re
import csv
import yaml
import sys
from pathlib import Path

def parse_filename(filename):
    """
    Parse a filename to extract source and language info.
    Expected format: <source>(<source-lang>-<target-lang>).txt
    """
    pattern = r"(.+)\((.+)-(.+)\)\.txt"
    match = re.match(pattern, filename)
    if not match:
        return None
    
    source_name = match.group(1)
    source_lang = match.group(2)
    target_lang = match.group(3)
    
    return {
        "source_name": source_name,
        "source_lang": source_lang,
        "target_lang": target_lang
    }

def get_all_languages():
    """Return all languages except English"""
    return ["es", "eu", "ga", "ca"]

def process_file(input_path, output_dir):
    """Process a single file and create corresponding directory and files"""
    filename = os.path.basename(input_path)
    file_info = parse_filename(filename)
    
    if not file_info:
        print(f"Skipping {filename}: Invalid format")
        return
    
    # Create experiment directory
    experiment_id = file_info["source_name"]
    experiment_dir = os.path.join(output_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Read phrases from the input file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            phrases = [line.strip() for line in f if line.strip()]
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        with open(input_path, 'r', encoding='latin-1') as f:
            phrases = [line.strip() for line in f if line.strip()]
    
    # Determine target languages
    target_langs = get_all_languages() if file_info["target_lang"] == "todos" else [file_info["target_lang"]]
    
    # Create experiment.yaml
    experiment_data = {
        "source_language": file_info["source_lang"],
        "target_languages": target_langs
    }
    
    with open(os.path.join(experiment_dir, "experiment.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(experiment_data, f, default_flow_style=False)
    
    # Create data.csv
    with open(os.path.join(experiment_dir, "data.csv"), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["source_phrase"])
        for phrase in phrases:
            writer.writerow([phrase])
    
    print(f"Processed {filename} -> {experiment_dir}")

def process_directory(input_dir, output_dir):
    """Process all .txt files in the input directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each .txt file
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_dir, filename)
            process_file(input_path, output_dir)

def create_query_script(output_dir):
    """Create a script in the output directory for loading and querying experiments"""
    script_content = '''
import os
import yaml
import csv
import pandas as pd

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

# Example usage:
# experiments = list_experiments("/path/to/data")
# print(experiments)
# experiment = load_experiment("books", "/path/to/data")
# print(experiment["config"])
# print(experiment["data"].head())
# filtered = query_experiment("books", "/path/to/data", "hello")
# print(filtered["data"])
'''
    
    with open(os.path.join(output_dir, "experiment_utils.py"), 'w', encoding='utf-8') as f:
        f.write(script_content)

def main(input_dir, output_dir):
    """Main function to process files and create query script"""
    process_directory(input_dir, output_dir)
    create_query_script(output_dir)
    print(f"Created query script at {os.path.join(output_dir, 'experiment_utils.py')}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    main(input_dir, output_dir)
