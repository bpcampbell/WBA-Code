import cv2
import numpy as np
from src.config import CONFIG

class OpticFlowGenerator:
    def __init__(self, frame_size, wavelength=50, amplitude=255):
        self.frame_size = frame_size
        self.wavelength = CONFIG['optic_flow'].get('wavelength', wavelength)
        self.amplitude = CONFIG['optic_flow'].get('amplitude', amplitude)
        
        # Pre-calculate allstatic components
        self.x, self.y = np.meshgrid(np.arange(frame_size[1]), np.arange(frame_size[0]))
        self.center_x = frame_size[1] // 2
        self.center_y = frame_size[0] // 2
        self.radius = np.sqrt((self.x - self.center_x)**2 + (self.y - self.center_y)**2)
        
        # Pre-calculate the spatial component
        self.spatial_term = 2 * np.pi * self.radius / self.wavelength
        
        # Pre-allocate arrays for better performance
        self.intensity = np.zeros(frame_size, dtype=np.uint8)
        self.intensity_3ch = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
        
        # Pre-calculate sin and cos terms
        self.sin_spatial = np.sin(self.spatial_term)
        self.cos_spatial = np.cos(self.spatial_term)
        
    def generate_pattern(self, phase):
        """Generate optic flow pattern for given phase."""
        # Use pre-calculated terms for faster computation
        intensity = self.amplitude * (
            self.sin_spatial * np.cos(phase) + 
            self.cos_spatial * np.sin(phase) + 1
        ) / 2
        
        # Avoid unnecessary array creation
        np.clip(intensity, 0, 255, out=self.intensity)
        
        # Avoid conversion, directly fill the 3-channel array
        self.intensity_3ch[..., 0] = self.intensity
        self.intensity_3ch[..., 1] = self.intensity
        self.intensity_3ch[..., 2] = self.intensity
        
        return self.intensity_3ch


