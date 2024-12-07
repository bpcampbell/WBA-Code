CONFIG = {
    # Wingbeat analyzer settings
    'wingbeat': {
        # Morphological operations
        'kernel_size': (6, 6),          # Operation kernel size
        'dilation_iterations': 2,        # Connect wing regions
        'erosion_iterations': 2,         # Clean noise
        
        # Detection thresholds
        'min_contour_area': 400,        # Minimum wing area (filters legs)
        'min_angle_threshold': 30,       # Minimum valid wing beat angle
        
        # Visualization
        'line_color': (255, 255, 0),    # Yellow lines
        'line_thickness': 1,            # Line thickness in pixels
    },
    
    # Optic flow settings
    'optic_flow': {
        'speed_scale': 500.0,          # Visual feedback scaling
        'wavelength': 50,               # Pattern wavelength
        'amplitude': 255                # Pattern contrast
    }
} 
