# scripts/run_training.py

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import TrainingConfig
from app.services.training_service import TrainingService

if __name__ == "__main__":
    config = TrainingConfig()
    training_service = TrainingService(config=config)

    print("Refactoring complete. Use this script to run training pipelines.")
