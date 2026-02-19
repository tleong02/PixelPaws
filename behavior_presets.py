"""
PixelPaws Behavior Presets
Predefined configurations for common behaviors based on scientific literature
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BehaviorPreset:
    """Configuration preset for a specific behavior."""
    name: str
    description: str
    bodyparts_brightness: List[str]
    min_bout_length: int  # frames
    max_gap: int  # frames
    square_size: int  # pixels
    recommended_species: List[str]
    citation: Optional[str] = None
    notes: Optional[str] = None


# Predefined behavior presets based on BAREfoot and literature
BEHAVIOR_PRESETS = {
    'flinching': BehaviorPreset(
        name='Flinching / Withdrawal',
        description='Rapid paw withdrawal response to painful stimulus',
        bodyparts_brightness=['hlpaw', 'hrpaw'],
        min_bout_length=4,
        max_gap=2,
        square_size=50,
        recommended_species=['mouse', 'rat'],
        citation='Barkai et al. (2025), Cell Rep Methods',
        notes='Paw elevation/brightness is critical. Typically 100-300ms duration.'
    ),
    
    'licking_biting': BehaviorPreset(
        name='Licking & Biting',
        description='Self-directed licking or biting of body parts (pain response)',
        bodyparts_brightness=['hlpaw', 'hrpaw', 'snout'],
        min_bout_length=10,
        max_gap=5,
        square_size=50,
        recommended_species=['mouse', 'rat'],
        citation='Barkai et al. (2025), Cell Rep Methods',
        notes='Snout-to-paw contact is key. Include snout brightness for detection.'
    ),
    
    'grooming': BehaviorPreset(
        name='Grooming',
        description='Self-grooming behavior (face washing, body grooming)',
        bodyparts_brightness=['hlpaw', 'hrpaw', 'snout'],
        min_bout_length=15,
        max_gap=10,
        square_size=50,
        recommended_species=['mouse', 'rat', 'hamster'],
        citation='Kalueff & Tuohimaa (2005), Nat Protoc',
        notes='Repetitive paw-to-face movements. Longer bouts than pain responses.'
    ),
    
    'scratching': BehaviorPreset(
        name='Scratching',
        description='Rapid scratching movements with hind paw',
        bodyparts_brightness=['hlpaw', 'hrpaw'],
        min_bout_length=8,
        max_gap=3,
        square_size=40,
        recommended_species=['mouse', 'rat'],
        notes='Rapid oscillating movements. Focus on hind paw elevation/velocity.'
    ),
    
    'rearing': BehaviorPreset(
        name='Rearing',
        description='Standing on hind legs (exploratory or anxiety behavior)',
        bodyparts_brightness=['hlpaw', 'hrpaw', 'flpaw', 'frpaw'],
        min_bout_length=10,
        max_gap=5,
        square_size=50,
        recommended_species=['mouse', 'rat'],
        citation='Prut & Belzung (2003), Eur J Pharmacol',
        notes='All paws lift from surface. Include forepaws for better detection.'
    ),
    
    'freezing': BehaviorPreset(
        name='Freezing',
        description='Complete immobility (fear/anxiety response)',
        bodyparts_brightness=['hlpaw', 'hrpaw', 'snout', 'tailbase'],
        min_bout_length=30,
        max_gap=10,
        square_size=50,
        recommended_species=['mouse', 'rat'],
        citation='Blanchard & Blanchard (1969), J Comp Physiol Psychol',
        notes='Lack of movement across all body parts. Longer bout durations typical.'
    ),
    
    'digging': BehaviorPreset(
        name='Digging',
        description='Digging behavior with forepaws',
        bodyparts_brightness=['flpaw', 'frpaw'],
        min_bout_length=20,
        max_gap=10,
        square_size=50,
        recommended_species=['mouse', 'rat', 'gerbil'],
        notes='Rapid alternating forepaw movements. Focus on forepaws only.'
    ),
    
    'jumping': BehaviorPreset(
        name='Jumping',
        description='Vertical jumping movements',
        bodyparts_brightness=['hlpaw', 'hrpaw', 'flpaw', 'frpaw', 'tailbase'],
        min_bout_length=5,
        max_gap=2,
        square_size=60,
        recommended_species=['mouse', 'rat'],
        notes='All body parts elevate. Brief duration. Large ROI for full body.'
    ),
    
    'tail_rattling': BehaviorPreset(
        name='Tail Rattling',
        description='Rapid tail vibration (aggression/arousal)',
        bodyparts_brightness=['tailbase'],
        min_bout_length=8,
        max_gap=3,
        square_size=40,
        recommended_species=['mouse', 'rat'],
        notes='Focus on tail movement. May not need paw brightness.'
    ),
    
    'sniffing': BehaviorPreset(
        name='Sniffing / Investigation',
        description='Active sniffing and investigation behavior',
        bodyparts_brightness=['snout'],
        min_bout_length=10,
        max_gap=5,
        square_size=40,
        recommended_species=['mouse', 'rat'],
        notes='Primarily snout movement. May combine with overall locomotion.'
    ),
    
    'custom': BehaviorPreset(
        name='Custom Behavior',
        description='User-defined behavior with manual settings',
        bodyparts_brightness=[],  # User will specify
        min_bout_length=10,
        max_gap=5,
        square_size=50,
        recommended_species=['all'],
        notes='Configure all parameters manually for your specific behavior.'
    )
}


def get_preset(behavior_type: str) -> BehaviorPreset:
    """
    Get preset configuration for a behavior type.
    
    Args:
        behavior_type: One of: 'flinching', 'licking_biting', 'grooming', 
                       'scratching', 'rearing', 'freezing', 'digging', 
                       'jumping', 'tail_rattling', 'sniffing', 'custom'
    
    Returns:
        BehaviorPreset with recommended settings
        
    Example:
        >>> preset = get_preset('flinching')
        >>> print(preset.bodyparts_brightness)
        ['hlpaw', 'hrpaw']
        >>> print(preset.min_bout_length)
        4
    """
    behavior_type = behavior_type.lower().replace(' ', '_')
    
    if behavior_type not in BEHAVIOR_PRESETS:
        available = list(BEHAVIOR_PRESETS.keys())
        raise ValueError(f"Unknown behavior type '{behavior_type}'. Available: {available}")
    
    return BEHAVIOR_PRESETS[behavior_type]


def get_all_presets() -> Dict[str, BehaviorPreset]:
    """Get all available behavior presets."""
    return BEHAVIOR_PRESETS.copy()


def get_preset_names() -> List[str]:
    """Get list of available preset names for dropdown menus."""
    return list(BEHAVIOR_PRESETS.keys())


def get_preset_display_names() -> List[str]:
    """Get formatted display names for UI dropdowns."""
    return [preset.name for preset in BEHAVIOR_PRESETS.values()]


def print_preset_info(behavior_type: str):
    """Print detailed information about a preset."""
    preset = get_preset(behavior_type)
    
    print(f"\n{'='*60}")
    print(f"Behavior: {preset.name}")
    print(f"{'='*60}")
    print(f"\nDescription:")
    print(f"  {preset.description}")
    print(f"\nRecommended Settings:")
    print(f"  Brightness body parts: {', '.join(preset.bodyparts_brightness)}")
    print(f"  Min bout length: {preset.min_bout_length} frames")
    print(f"  Max gap: {preset.max_gap} frames")
    print(f"  Square size: {preset.square_size} pixels")
    print(f"\nSpecies:")
    print(f"  {', '.join(preset.recommended_species)}")
    
    if preset.citation:
        print(f"\nCitation:")
        print(f"  {preset.citation}")
    
    if preset.notes:
        print(f"\nNotes:")
        print(f"  {preset.notes}")
    print(f"{'='*60}\n")


def compare_presets(behavior_types: List[str]):
    """Compare multiple behavior presets side by side."""
    presets = [get_preset(bt) for bt in behavior_types]
    
    print(f"\n{'Comparison':<20}", end='')
    for preset in presets:
        print(f"{preset.name:<25}", end='')
    print()
    print("="*100)
    
    # Brightness body parts
    print(f"{'Brightness parts':<20}", end='')
    for preset in presets:
        parts_str = ', '.join(preset.bodyparts_brightness[:2]) + ('...' if len(preset.bodyparts_brightness) > 2 else '')
        print(f"{parts_str:<25}", end='')
    print()
    
    # Min bout length
    print(f"{'Min bout (frames)':<20}", end='')
    for preset in presets:
        print(f"{preset.min_bout_length:<25}", end='')
    print()
    
    # Max gap
    print(f"{'Max gap (frames)':<20}", end='')
    for preset in presets:
        print(f"{preset.max_gap:<25}", end='')
    print()
    
    # Square size
    print(f"{'Square size (px)':<20}", end='')
    for preset in presets:
        print(f"{preset.square_size:<25}", end='')
    print()
    
    print("="*100)


# GUI-ready data structures
GUI_PRESET_OPTIONS = [
    ("Flinching / Withdrawal", "flinching"),
    ("Licking & Biting", "licking_biting"),
    ("Grooming", "grooming"),
    ("Scratching", "scratching"),
    ("Rearing", "rearing"),
    ("Freezing", "freezing"),
    ("Digging", "digging"),
    ("Jumping", "jumping"),
    ("Tail Rattling", "tail_rattling"),
    ("Sniffing / Investigation", "sniffing"),
    ("Custom (Manual Settings)", "custom")
]


def get_gui_options():
    """Get options formatted for GUI dropdown menus."""
    return GUI_PRESET_OPTIONS


if __name__ == "__main__":
    print("PixelPaws Behavior Presets")
    print("="*60)
    print("\nAvailable presets:")
    for i, (display_name, key) in enumerate(GUI_PRESET_OPTIONS, 1):
        print(f"  {i}. {display_name}")
    
    print("\n" + "="*60)
    print("Example Usage:")
    print("="*60)
    
    # Example 1: Get preset
    print("\n1. Get preset configuration:")
    print("   >>> preset = get_preset('flinching')")
    print("   >>> preset.bodyparts_brightness")
    preset = get_preset('flinching')
    print(f"   {preset.bodyparts_brightness}")
    
    # Example 2: Use in training
    print("\n2. Use preset in training:")
    print("   >>> preset = get_preset('grooming')")
    print("   >>> features = extract_all_features(")
    print("   ...     dlc_file='video_DLC.h5',")
    print("   ...     video_file='video.mp4',")
    print("   ...     bodyparts=all_bodyparts,  # Use all for pose")
    print("   ...     bodyparts_brightness=preset.bodyparts_brightness,")
    print("   ...     square_size=preset.square_size")
    print("   ... )")
    
    # Example 3: Print detailed info
    print("\n3. View detailed preset information:")
    print_preset_info('flinching')
    
    # Example 4: Compare presets
    print("\n4. Compare multiple behaviors:")
    compare_presets(['flinching', 'grooming', 'rearing'])
