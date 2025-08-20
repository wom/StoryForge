"""
Configuration management for StoryForge.

Handles loading, parsing, and merging configuration from:
1. Configuration files (INI format)
2. Environment variables
3. Command line arguments (highest priority)

Configuration file priority:
1. STORYFORGE_CONFIG environment variable path
2. XDG config directory: ~/.config/storyforge/storyforge.ini
3. Home directory: ~/.storyforge.ini
4. Current directory: ./storyforge.ini
"""

import os
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Dict, List, Optional

from platformdirs import user_config_dir
from rich.console import Console

console = Console()

# Default configuration template
DEFAULT_CONFIG_TEMPLATE = """# StoryForge Configuration File
# This file contains default values for story generation parameters
# Command line arguments will override these settings

[story]
# Story length options: flash, short, medium, bedtime
length = bedtime

# Target age group options: toddler, preschool, early_reader, middle_grade  
age_range = early_reader

# Story style options: adventure, comedy, fantasy, fairy_tale, friendship, random
style = random

# Story tone options: gentle, exciting, silly, heartwarming, magical, random
tone = random

# Story theme options: courage, kindness, teamwork, problem_solving, creativity, random
theme = random

# Learning focus options: counting, colors, letters, emotions, nature (leave empty for none)
learning_focus = 

# Default story setting (leave empty for none)
setting = 

# Default characters, comma-separated (leave empty for none)
characters = 

[images]
# Image art style options: chibi, realistic, cartoon, watercolor, sketch
image_style = chibi

[output]
# Default output directory (leave empty for auto-generated timestamp)
output_dir = 

# Whether to use context files by default: true, false
use_context = true

[system]
# LLM backend options: gemini, openai, anthropic (leave empty for auto-detection)
# This overrides the LLM_BACKEND environment variable if set
backend = 

# Enable verbose output by default: true, false
verbose = false

# Enable debug mode by default: true, false  
debug = false
"""

# Valid configuration values for validation
VALID_VALUES = {
    'length': ['flash', 'short', 'medium', 'bedtime'],
    'age_range': ['toddler', 'preschool', 'early_reader', 'middle_grade'],
    'style': ['adventure', 'comedy', 'fantasy', 'fairy_tale', 'friendship', 'random'],
    'tone': ['gentle', 'exciting', 'silly', 'heartwarming', 'magical', 'random'],
    'theme': ['courage', 'kindness', 'teamwork', 'problem_solving', 'creativity', 'random'],
    'learning_focus': ['counting', 'colors', 'letters', 'emotions', 'nature', ''],
    'image_style': ['chibi', 'realistic', 'cartoon', 'watercolor', 'sketch'],
    'backend': ['gemini', 'openai', 'anthropic', ''],
}


class ConfigError(Exception):
    """Configuration related errors."""
    pass


class Config:
    """Configuration manager for StoryForge."""
    
    def __init__(self):
        self.config = ConfigParser()
        self.config_path: Optional[Path] = None
        self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration values."""
        self.config.read_string(DEFAULT_CONFIG_TEMPLATE)
    
    def get_config_paths(self) -> List[Path]:
        """Return configuration file paths in priority order."""
        paths = []
        
        # 1. STORYFORGE_CONFIG environment variable (highest priority)
        env_config = os.environ.get("STORYFORGE_CONFIG")
        if env_config:
            paths.append(Path(env_config))
        
        # 2. XDG config directory  
        paths.append(Path(user_config_dir("storyforge", "StoryForge")) / "storyforge.ini")
        
        # 3. Home directory fallback
        paths.append(Path.home() / ".storyforge.ini")
        
        # 4. Current directory fallback
        paths.append(Path("./storyforge.ini"))
        
        return paths
    
    def find_config_file(self) -> Optional[Path]:
        """Find the first existing configuration file."""
        for path in self.get_config_paths():
            if path.exists() and path.is_file():
                return path
        return None
    
    def load_config(self, verbose: bool = False) -> bool:
        """
        Load configuration from file.
        
        Returns:
            bool: True if config file was found and loaded, False otherwise.
        """
        config_path = self.find_config_file()
        if not config_path:
            if verbose:
                console.print("[dim]No configuration file found, using defaults[/dim]")
            return False
        
        try:
            self.config.read(config_path)
            self.config_path = config_path
            if verbose:
                console.print(f"[dim]Loaded configuration from: {config_path}[/dim]")
            return True
        except Exception as e:
            raise ConfigError(f"Error reading configuration file {config_path}: {e}") from e
    
    def validate_config(self) -> List[str]:
        """
        Validate configuration values.
        
        Returns:
            List[str]: List of validation errors, empty if valid.
        """
        errors = []
        
        for section_name in self.config.sections():
            section = self.config[section_name]
            
            for key, value in section.items():
                # Skip comment keys
                if key.startswith('#'):
                    continue
                
                # Check if we have validation rules for this key
                if key in VALID_VALUES:
                    valid_options = VALID_VALUES[key]
                    if value and value not in valid_options:
                        errors.append(
                            f"Invalid value '{value}' for {key}. "
                            f"Valid options: {', '.join(valid_options)}"
                        )
        
        return errors
    
    def get_default_config_path(self) -> Path:
        """Get the default configuration file path (XDG config directory)."""
        return Path(user_config_dir("storyforge", "StoryForge")) / "storyforge.ini"
    
    def create_default_config(self, path: Optional[Path] = None) -> Path:
        """
        Create a default configuration file.
        
        Args:
            path: Path to create config file. If None, uses default location.
            
        Returns:
            Path: The path where the config file was created.
        """
        if path is None:
            path = self.get_default_config_path()
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write default configuration
        with open(path, 'w', encoding='utf-8') as f:
            f.write(DEFAULT_CONFIG_TEMPLATE)
        
        return path
    
    def get_value(self, section: str, key: str, fallback: str = "") -> str:
        """Get configuration value with fallback."""
        try:
            value = self.config.get(section, key, fallback=fallback)
            return value.strip() if value else ""
        except Exception:
            return fallback
    
    def get_bool(self, section: str, key: str, fallback: bool = False) -> bool:
        """Get boolean configuration value with fallback."""
        try:
            return self.config.getboolean(section, key, fallback=fallback)
        except Exception:
            return fallback
    
    def get_list(self, section: str, key: str, fallback: Optional[List[str]] = None) -> Optional[List[str]]:
        """Get list configuration value (comma-separated) with fallback."""
        if fallback is None:
            fallback = []
        
        try:
            value = self.get_value(section, key)
            if not value:
                return None if not fallback else fallback
            
            # Split by comma and strip whitespace
            return [item.strip() for item in value.split(',') if item.strip()]
        except Exception:
            return fallback if fallback else None
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert configuration to dictionary format."""
        result = {}
        for section_name in self.config.sections():
            result[section_name] = {}
            for key, value in self.config[section_name].items():
                # Skip comment keys
                if not key.startswith('#'):
                    result[section_name][key] = value
        return result


def load_config(verbose: bool = False) -> Config:
    """
    Load configuration from file system.
    
    Args:
        verbose: Enable verbose output
        
    Returns:
        Config: Loaded configuration object
        
    Raises:
        ConfigError: If configuration file is malformed
    """
    config = Config()
    config.load_config(verbose=verbose)
    
    # Validate configuration
    errors = config.validate_config()
    if errors:
        raise ConfigError(
            f"Configuration validation failed:\n" + 
            "\n".join(f"  - {error}" for error in errors)
        )
    
    return config