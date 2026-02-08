"""
=============================================================================
 EXPERIMENT LOGGER - Hyperparameter & Training Tracking
 Logs experiments for reproducibility and comparison
=============================================================================
"""

import json
import os
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """Logger for ML experiments with hyperparameters and results."""

    def __init__(self, experiment_name=None, log_dir="training/logs"):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of experiment (auto-generated if None)
            log_dir: Directory to save logs
        """
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"

        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_file = self.log_dir / f"{experiment_name}.json"

        self.data = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'hyperparameters': {},
            'training_config': {},
            'epoch_logs': [],
            'final_metrics': {},
            'model_path': None,
            'status': 'initialized'
        }

    def log_hyperparameters(self, **kwargs):
        """Log hyperparameters (learning rate, batch size, etc.)"""
        self.data['hyperparameters'].update(kwargs)
        self._save()
        print(f"[LOGGER] Hyperparameters logged: {list(kwargs.keys())}")

    def log_config(self, **kwargs):
        """Log training configuration (device, dataset path, etc.)"""
        self.data['training_config'].update(kwargs)
        self._save()
        print(f"[LOGGER] Training config logged: {list(kwargs.keys())}")

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, time_elapsed):
        """Log metrics for a single epoch."""
        epoch_data = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'learning_rate': float(lr),
            'time_seconds': float(time_elapsed)
        }
        self.data['epoch_logs'].append(epoch_data)
        self._save()

    def log_final_metrics(self, metrics_dict):
        """Log final evaluation metrics."""
        self.data['final_metrics'] = metrics_dict
        self._save()
        print(f"[LOGGER] Final metrics logged")

    def log_model_path(self, model_path):
        """Log the path to saved model."""
        self.data['model_path'] = str(model_path)
        self._save()

    def set_status(self, status):
        """Set experiment status (initialized, running, completed, failed)."""
        self.data['status'] = status
        self.data['last_updated'] = datetime.now().isoformat()
        self._save()

    def _save(self):
        """Save experiment data to JSON file."""
        with open(self.experiment_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get_summary(self):
        """Get a summary of the experiment."""
        if not self.data['epoch_logs']:
            return "No training data logged yet."

        best_val_acc = max(log['val_acc'] for log in self.data['epoch_logs'])
        total_epochs = len(self.data['epoch_logs'])

        summary = f"""
        ========================================
        Experiment: {self.experiment_name}
        ========================================
        Status: {self.data['status']}
        Total Epochs: {total_epochs}
        Best Val Accuracy: {best_val_acc:.2f}%

        Hyperparameters:
        {json.dumps(self.data['hyperparameters'], indent=2)}

        Final Metrics:
        {json.dumps(self.data['final_metrics'], indent=2)}
        ========================================
        """
        return summary

    def __repr__(self):
        return f"ExperimentLogger(name='{self.experiment_name}', status='{self.data['status']}')"


def load_experiment(experiment_file):
    """
    Load a previous experiment from JSON file.

    Args:
        experiment_file: Path to experiment JSON file

    Returns:
        dict: Experiment data
    """
    with open(experiment_file, 'r') as f:
        return json.load(f)


def list_experiments(log_dir="training/logs"):
    """
    List all experiments in log directory.

    Args:
        log_dir: Directory containing experiment logs

    Returns:
        list: List of experiment filenames
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return []

    experiments = list(log_path.glob("experiment_*.json"))
    return [exp.stem for exp in experiments]
