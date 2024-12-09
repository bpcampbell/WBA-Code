CONFIG = {
    # Experiment settings
    'experiment': {
        'name': 'realtime_test',
        'frame_size': (800, 800),
        'gain_cycle': [
            {'duration': 10, 'gain': 50},  # Higher gain for more noticeable movement
            {'duration': 10, 'gain': 0},   # No stimulus
        ],
        'min_amplitude': 0,
        'max_amplitude': 180
    },
    
    # Wingbeat analyzer settings
    'wingbeat': {
        # Morphological operations
        'kernel_size': (6, 6),          # Operation kernel size
        'dilation_iterations': 2,        # Connect wing regions
        'erosion_iterations': 2,         # Clean noise
        
        # Detection thresholds
        'min_contour_area': 800,        # Minimum wing area (filters legs)
        'min_angle_threshold': 50,       # Minimum valid wing beat angle
        'max_angle_threshold': 180,      # Maximum valid wing beat angle
        'min_wing_length': 20,          # Minimum wing length
        
        # Visualization
        'line_color': (255, 255, 0),    # Yellow lines
        'line_thickness': 1,            # Line thickness in pixels
    },
    
    # Optic flow settings
    'optic_flow': {
        'speed_scale': 1.0,          # Visual feedback scaling
        'wavelength': 50,               # Pattern wavelength
        'amplitude': 255                # Pattern contrast
    }
}