"""
Centralized Data Path Configuration
====================================

Provides a unified interface for accessing data files in both:
1. Legacy flat structure: results/paper_data/aes_FULLGRID_*.npz
2. Experiment structure: results/paper_data/experiments/<experiment_name>/<experiment_name>_source_*.npz

Usage:
------
    from research.data_config import DataConfig
    
    # For legacy mode (default)
    config = DataConfig()
    
    # For experiment mode
    config = DataConfig(
        mode="experiment",
        experiment_name="aes_fullgrid_perpair_srcgrid3x5_corner2.0_per_pair"
    )
    
    # Get data directory
    data_dir = config.get_data_dir()
    
    # Get list of files to process
    files = config.get_files()
    
    # Extract source tag from filename
    source_tag = config.extract_source_tag(filename)
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Literal
from pathlib import Path


class DataConfig:
    """Centralized configuration for data file paths and naming conventions."""
    
    def __init__(
        self,
        mode: Literal["legacy", "experiment"] = "legacy",
        experiment_name: Optional[str] = None,
        legacy_files: Optional[List[str]] = None,
        project_root: Optional[str] = None,
    ):
        """
        Initialize DataConfig.
        
        Args:
            mode: "legacy" for flat structure, "experiment" for experiment directories
            experiment_name: Name of experiment directory (required if mode="experiment")
            legacy_files: List of specific legacy files to process (optional)
            project_root: Override project root path (auto-detected if None)
        """
        self.mode = mode
        self.experiment_name = experiment_name
        self._legacy_files = legacy_files
        
        # Auto-detect project root if not provided
        if project_root is None:
            # Assume this file is in research/ directory
            self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.project_root = project_root
        
        # Validate configuration
        if self.mode == "experiment" and not self.experiment_name:
            raise ValueError("experiment_name is required when mode='experiment'")
    
    def get_data_dir(self) -> str:
        """
        Get the data directory path based on current mode.
        
        Returns:
            Absolute path to data directory
        """
        base_dir = os.path.join(self.project_root, "results", "paper_data")
        
        if self.mode == "legacy":
            return base_dir
        else:  # experiment mode
            return os.path.join(base_dir, "experiments", self.experiment_name)
    
    def get_files(self, num_sources: Optional[int] = None) -> List[str]:
        """
        Get list of files to process based on current mode.
        
        Args:
            num_sources: For experiment mode, number of source files to include.
                        If None, auto-detects all available source files.
        
        Returns:
            List of filenames (not full paths, just basenames)
        """
        if self.mode == "legacy":
            # Return user-specified files or default legacy files
            if self._legacy_files:
                return self._legacy_files
            else:
                # Default legacy files
                return [
                    "aes_FULLGRID_center_source.npz",
                    "aes_FULLGRID_top_middle_source.npz",
                    "aes_FULLGRID_upper_right_source.npz",
                    "aes_FULLGRID_lower_left_source.npz",
                    "aes_FULLGRID_corner_sourcev3.npz",
                ]
        else:  # experiment mode
            if num_sources is None:
                # Auto-detect by scanning directory
                num_sources = self._detect_num_sources()
            
            return [
                f"{self.experiment_name}_source_{i}.npz"
                for i in range(1, num_sources + 1)
            ]
    
    def _detect_num_sources(self) -> int:
        """
        Auto-detect number of source files in experiment directory.
        
        Returns:
            Number of source files found
        """
        data_dir = self.get_data_dir()
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Experiment directory not found: {data_dir}")
        
        # Pattern: <experiment_name>_source_<N>.npz
        pattern = re.compile(rf"{re.escape(self.experiment_name)}_source_(\d+)\.npz$")
        
        source_numbers = []
        for filename in os.listdir(data_dir):
            match = pattern.match(filename)
            if match:
                source_numbers.append(int(match.group(1)))
        
        if not source_numbers:
            raise ValueError(f"No source files found in {data_dir}")
        
        return max(source_numbers)
    
    def extract_source_tag(self, filename: str) -> str:
        """
        Extract a human-readable source tag from filename.
        
        Args:
            filename: Filename to extract tag from
        
        Returns:
            Source tag string (e.g., "center_source", "source_1")
        
        Examples:
            Legacy: "aes_FULLGRID_center_source.npz" -> "center_source"
            Experiment: "aes_fullgrid_perpair_..._source_1.npz" -> "source_1"
        """
        if self.mode == "legacy":
            # Remove prefix and suffix
            # "aes_FULLGRID_center_source.npz" -> "center_source"
            tag = filename.replace("aes_FULLGRID_", "").replace("aes_quarter_", "")
            tag = tag.replace("cube6_FULLGRID_", "").replace(".npz", "")
            return tag
        else:  # experiment mode
            # Extract source number from pattern: ..._source_N.npz
            match = re.search(r'_source_(\d+)\.npz$', filename)
            if match:
                return f"source_{match.group(1)}"
            else:
                # Fallback: just remove .npz
                return filename.replace(".npz", "")
    
    def extract_source_name_for_display(self, filename: str) -> str:
        """
        Extract a display-friendly source name from filename.
        
        Args:
            filename: Filename to extract name from
        
        Returns:
            Display name (e.g., "Center Source", "Source 1")
        """
        tag = self.extract_source_tag(filename)
        
        if self.mode == "legacy":
            # "center_source" -> "Center Source"
            return tag.replace("_", " ").title()
        else:
            # "source_1" -> "Source 1"
            return tag.replace("_", " ").title()
    
    def is_experiment_mode(self) -> bool:
        """Check if currently in experiment mode."""
        return self.mode == "experiment"
    
    def get_full_path(self, filename: str) -> str:
        """
        Get full absolute path for a data file.
        
        Args:
            filename: Filename (basename)
        
        Returns:
            Absolute path to file
        """
        return os.path.join(self.get_data_dir(), filename)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        if self.mode == "legacy":
            return f"DataConfig(mode='legacy', num_files={len(self.get_files())})"
        else:
            return f"DataConfig(mode='experiment', experiment='{self.experiment_name}')"


# Convenience factory functions
def create_legacy_config(files: Optional[List[str]] = None) -> DataConfig:
    """
    Create a DataConfig for legacy flat file structure.
    
    Args:
        files: Optional list of specific files to process
    
    Returns:
        DataConfig instance
    """
    return DataConfig(mode="legacy", legacy_files=files)


def create_experiment_config(experiment_name: str, num_sources: Optional[int] = None) -> DataConfig:
    """
    Create a DataConfig for experiment directory structure.
    
    Args:
        experiment_name: Name of experiment directory
        num_sources: Number of source files (auto-detected if None)
    
    Returns:
        DataConfig instance
    """
    config = DataConfig(mode="experiment", experiment_name=experiment_name)
    # Validate by trying to get files
    config.get_files(num_sources=num_sources)
    return config


# Example usage and testing
if __name__ == "__main__":
    print("=== DataConfig Examples ===\n")
    
    # Legacy mode
    print("1. Legacy mode (default):")
    legacy = DataConfig()
    print(f"   {legacy}")
    print(f"   Data dir: {legacy.get_data_dir()}")
    print(f"   Files: {legacy.get_files()[:2]}...")
    print(f"   Source tag: {legacy.extract_source_tag('aes_FULLGRID_center_source.npz')}")
    print()
    
    # Experiment mode
    print("2. Experiment mode:")
    try:
        experiment = DataConfig(
            mode="experiment",
            experiment_name="aes_fullgrid_perpair_srcgrid3x5_corner2.0_per_pair"
        )
        print(f"   {experiment}")
        print(f"   Data dir: {experiment.get_data_dir()}")
        files = experiment.get_files()
        print(f"   Files ({len(files)} total): {files[:3]}...")
        print(f"   Source tag: {experiment.extract_source_tag(files[0])}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Custom legacy files
    print("3. Legacy mode with custom files:")
    custom = create_legacy_config(files=["aes_FULLGRID_center_source.npz"])
    print(f"   {custom}")
    print(f"   Files: {custom.get_files()}")
