"""
Book Writer System - Configuration Module
Manages configuration for model selection and settings
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ModelConfig:
    """Configuration class for managing AI model settings."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the model configuration.
        
        Args:
            config_path: Path to the configuration file (optional)
        """
        self.config_path = config_path or Path(__file__).parent / "model_config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default configuration."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            # Create default configuration
            default_config = {
                "models": {
                    "content_expansion": {
                        "model_name": "stable-beluga:13b",
                        "api_type": "ollama",
                        "temperature": 0.7,
                        "max_tokens": 1024,
                        "top_p": 0.9
                    },
                    "outline_generation": {
                        "model_name": "deepseek-r1:8b",
                        "api_type": "ollama",
                        "temperature": 0.5,
                        "max_tokens": 512,
                        "top_p": 0.8
                    },
                    "organization": {
                        "model_name": "deepseek-r1:8b",
                        "api_type": "ollama",
                        "temperature": 0.4,
                        "max_tokens": 512,
                        "top_p": 0.8
                    }
                },
                "ollama": {
                    "base_url": "http://localhost:11434"
                }
            }
            
            # Create the config file with defaults
            self._save_config(default_config)
            return default_config
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def get_model_config(self, task: str) -> Dict[str, Any]:
        """Get configuration for a specific task.
        
        Args:
            task: The task name (e.g., 'content_expansion', 'outline_generation', 'organization')
            
        Returns:
            Configuration dictionary for the specified task
        """
        if task in self.config["models"]:
            return self.config["models"][task]
        else:
            # Return a default configuration if task not found
            return {
                "model_name": "stable-beluga:13b",
                "api_type": "ollama",
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 0.9
            }
    
    def get_ollama_config(self) -> Dict[str, Any]:
        """Get Ollama-specific configuration.
        
        Returns:
            Ollama configuration dictionary
        """
        return self.config.get("ollama", {
            "base_url": "http://localhost:11434"
        })
    
    def update_model_config(self, task: str, config: Dict[str, Any]) -> None:
        """Update configuration for a specific task.
        
        Args:
            task: The task name
            config: New configuration for the task
        """
        if "models" not in self.config:
            self.config["models"] = {}
        
        self.config["models"][task] = config
        self._save_config(self.config)


# Global model configuration instance
model_config = ModelConfig()