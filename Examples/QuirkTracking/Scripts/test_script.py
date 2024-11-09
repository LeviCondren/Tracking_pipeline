import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
print(f"Script directory: {script_dir}")

# Construct the path to the project root
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
print(f"Project root: {project_root}")

# Add the project root to the Python path
sys.path.append(project_root)
print(f"sys.path: {sys.path}")

# Check if the project root directory is in sys.path
if project_root not in sys.path:
    print(f"Error: {project_root} is not in sys.path")
else:
    print(f"Success: {project_root} is in sys.path")

# Verify if the Pipelines directory exists in the project root
pipelines_path = os.path.join(project_root, "Pipelines")
if os.path.isdir(pipelines_path):
    print(f"Success: {pipelines_path} exists")
else:
    print(f"Error: {pipelines_path} does not exist")

# Try importing the modules
try:
    from Pipelines.TrackML_Example.LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding
    from Pipelines.TrackML_Example.LightningModules.Processing.feature_store_base import FeatureStoreBase
    print("Imports successful")
except ImportError as e:
    print(f"Import error: {e}")
