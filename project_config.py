"""
project_config.py — Centralised Project Configuration
======================================================
Single data model for PixelPaws_project.json, used by:
  - PixelPaws_GUI.py (save_project_config, _load_project_config)
  - project_setup.py  (_save_step2_config, wizard finish)

Provides load/save with merge semantics (new values overwrite, missing
keys are preserved from disk) and safe defaults.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional


CONFIG_FILENAME = 'PixelPaws_project.json'


@dataclass
class ProjectConfig:
    """In-memory representation of a PixelPaws project config.

    Fields mirror the keys written to PixelPaws_project.json.  Not every
    field is always present — ``load()`` fills gaps with safe defaults.
    """
    project_folder: str = ''
    video_ext: str = '.avi'
    behaviors: List[str] = field(default_factory=list)
    behavior_name: str = ''
    bp_include_list: Optional[List[str]] = None
    bp_pixbrt_list: List[str] = field(default_factory=list)
    square_size: List[int] = field(default_factory=lambda: [40])
    pix_threshold: float = 0.3
    include_optical_flow: bool = False
    bp_optflow_list: List[str] = field(default_factory=list)
    roi_size: int = 20
    dlc_config: str = ''
    last_classifier: str = ''

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, project_folder: str) -> 'ProjectConfig':
        """Load config from ``<project_folder>/PixelPaws_project.json``.

        Returns a ``ProjectConfig`` with safe defaults for any missing keys.
        Never raises — returns defaults on any I/O or parse error.
        """
        cfg = cls(project_folder=project_folder)
        config_path = os.path.join(project_folder, CONFIG_FILENAME)
        if not os.path.isfile(config_path):
            return cfg
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: could not load project config {config_path}: {e}")
            return cfg

        # Map JSON keys → dataclass fields (same names)
        for key in (
            'project_folder', 'video_ext', 'behaviors', 'behavior_name',
            'bp_include_list', 'bp_pixbrt_list', 'square_size',
            'pix_threshold', 'include_optical_flow', 'bp_optflow_list',
            'roi_size', 'dlc_config', 'last_classifier',
        ):
            if key in data:
                setattr(cfg, key, data[key])

        # Legacy: roi_size → square_size
        if 'square_size' not in data and 'roi_size' in data:
            rs = data['roi_size']
            cfg.square_size = [int(rs)] if not isinstance(rs, list) else [int(x) for x in rs]

        return cfg

    # ------------------------------------------------------------------
    # Save (merge semantics)
    # ------------------------------------------------------------------

    def save(self, project_folder: str = None) -> None:
        """Save config to ``<folder>/PixelPaws_project.json``.

        Uses merge semantics: loads existing file first, then overlays
        non-empty values from this instance.  This preserves any keys
        that this dataclass doesn't model.
        """
        folder = project_folder or self.project_folder
        if not folder or not os.path.isdir(folder):
            return

        config_path = os.path.join(folder, CONFIG_FILENAME)

        # Load existing to preserve unknown keys
        existing = {}
        if os.path.isfile(config_path):
            try:
                with open(config_path, 'r') as f:
                    existing = json.load(f)
            except Exception:
                pass

        # Overlay non-default values
        updates = asdict(self)
        updates['project_folder'] = folder
        for k, v in list(updates.items()):
            # Skip None / empty-string / empty-list when existing has a value
            if v is None or v == '' or v == []:
                if k in existing:
                    continue
            existing[k] = v

        try:
            os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            print(f"Warning: could not save project config: {e}")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain dict (for passing to hash functions etc.)."""
        return asdict(self)
