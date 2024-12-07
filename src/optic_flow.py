import cv2
import numpy as np
from src.config import CONFIG

class OpticFlowGenerator:
    def __init__(self, frame_size, wavelength=50, amplitude=255):
        self.frame_size = frame_size
        self.wavelength = CONFIG['optic_flow'].get('wavelength', wavelength)
        self.amplitude = CONFIG['optic_flow'].get('amplitude', amplitude)
        
        # Pre-calculate static components
        self.x, self.y = np.meshgrid(np.arange(frame_size[1]), np.arange(frame_size[0]))
        self.center_x = frame_size[1] // 2
        self.center_y = frame_size[0] // 2
        self.radius = np.sqrt((self.x - self.center_x)**2 + (self.y - self.center_y)**2)
        
    def generate_pattern(self, phase):
        """Generate optic flow pattern for given phase."""
        # Calculate intensity
        intensity = self.amplitude * (np.sin(2 * np.pi * self.radius / self.wavelength + phase) + 1) / 2
        intensity = intensity.astype(np.uint8)
        intensity_3ch = cv2.cvtColor(intensity[:, :, np.newaxis], cv2.COLOR_GRAY2BGR)
        
        # Calculate flow field
        flow_x = np.cos(2 * np.pi * self.radius / self.wavelength + phase)
        flow_y = np.sin(2 * np.pi * self.radius / self.wavelength + phase)
        
        # Normalize flow fields
        flow_x = cv2.normalize(flow_x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        flow_y = cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        
        return cv2.merge([intensity_3ch, flow_x, flow_y])[:, :, :3]
