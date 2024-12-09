import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import Tuple, Any
from src.video_handler import VideoHandler
from src.data_manager import DataManager
from src.config import CONFIG
from datetime import datetime
import customtkinter as ctk
from PIL import Image, ImageTk
from src.point_selector import PointSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WingbeatAnalyzer:
    def __init__(self, config):
        self.config = config
        self.points = None  # Will be set externally
        self.setup_masks()
        self.setup_points()
        self.last_cropped_frame = None
        
    def setup_masks(self):
        """Initialize mask parameters"""
        self.mask = None
        self.mask2 = None
        self.mask_x1 = None
        self.mask_x2 = None
        self.distance = None
        self.square_size = None
        
    def setup_points(self):
        """Initialize transformed points"""
        self.central_point_new = None
        self.left_point_new = None
        self.right_point_new = None
        
    def initialize_processing(self, frame):
        """Initialize processing parameters from first frame"""
        if self.points is None:
            raise ValueError("Points must be set before initialization")
            
        central_point = (self.points['x0'], self.points['y0'])
        head_point = (self.points['x1'], self.points['y1'])
        wing_point = (self.points['x2'], self.points['y2'])
        left_hinge = (self.points['x3'], self.points['y3'])
        right_hinge = (self.points['x4'], self.points['y4'])
        
        # Calculate distance and square region
        self.distance = int(np.sqrt((wing_point[0] - central_point[0]) ** 2 + 
                                  (wing_point[1] - central_point[1]) ** 2))
        self.square_size = 2 * self.distance
        
        # Convert frame to grayscale if needed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) > 2 else frame
        
        # Sample regions around points
        region_size = 5  # Size of sampling region
        
        # Sample body region (center and head)
        body_samples = []
        for point in [central_point, head_point]:
            x, y = point
            region = gray[y-region_size:y+region_size, x-region_size:x+region_size]
            body_samples.extend(region.flatten())
        body_mean = np.mean(body_samples)
        body_std = np.std(body_samples)
        
        # Sample wing regions (wing point and hinges)
        wing_samples = []
        for point in [wing_point, left_hinge, right_hinge]:
            x, y = point
            region = gray[y-region_size:y+region_size, x-region_size:x+region_size]
            wing_samples.extend(region.flatten())
        wing_mean = np.mean(wing_samples)
        wing_std = np.std(wing_samples)
        
        # Update thresholds based on sampled values
        self.config['intensity_thresholds'] = {
            'body': max(body_mean - body_std, wing_mean + wing_std),  # Body threshold
            'wing_min': max(10, wing_mean - wing_std),  # Lower wing bound
            'wing_max': min(wing_mean + wing_std, body_mean - body_std)  # Upper wing bound
        }
        
        logger.info(f"Calculated thresholds: {self.config['intensity_thresholds']}")
        
        # Continue with point transformation and mask initialization
        self._calculate_transformed_points(central_point, left_hinge, right_hinge)
        self._initialize_masks(frame)
        
    def _calculate_transformed_points(self, central_point, left_point, right_point):
        """Calculate transformed points for the cropped region"""
        # Central point
        self.central_point_new = (int(self.square_size / 2), int(self.square_size / 2))
        
        # Left point
        left_point_new_x = int((self.square_size / 2) - (central_point[0] - left_point[0]))
        left_point_new_y = int((self.square_size / 2) - (central_point[1] - left_point[1]))
        self.left_point_new = (left_point_new_x, left_point_new_y)
        
        # Right point
        right_point_new_x = int((self.square_size / 2) + (right_point[0] - central_point[0]))
        right_point_new_y = int((self.square_size / 2) - (central_point[1] - right_point[1]))
        self.right_point_new = (right_point_new_x, right_point_new_y)
        
    def _initialize_masks(self, frame):
        """Initialize binary masks"""
        # Pre-allocate arrays we'll use repeatedly
        self.mask = np.zeros((self.square_size, self.square_size), dtype="uint8")
        self.mask2 = np.zeros_like(self.mask)
        self.binary_frame = np.zeros_like(self.mask)
        self.temp_roi = np.zeros_like(self.mask)  # For contour operations
        
        # Calculate mask boundaries
        self.mask_x1 = int((self.square_size / 2) - (self.central_point_new[0] - self.left_point_new[0]))
        self.mask_x2 = int((self.square_size / 2) + (self.right_point_new[0] - self.central_point_new[0]))
        
        # Create primary mask
        self.mask = np.zeros((self.square_size, self.square_size), dtype="uint8")
        cv2.rectangle(self.mask, (self.mask_x1, 0), (self.mask_x2, self.square_size), 255, -1)
        
        # Create radius masks
        radius_mask = int(self.distance - 2*self.distance/10)
        cv2.circle(self.mask, self.central_point_new, radius_mask, 255, -1)
        
        # Create second mask
        self.mask2 = np.zeros((self.square_size, self.square_size), dtype="uint8")
        radius_mask_2 = int(self.distance - self.distance/10)
        cv2.circle(self.mask2, self.central_point_new, radius_mask_2, 255, -1)
        
    def crop_frame(self, frame):
        """Crop frame to region of interest"""
        x_square = self.points['x0'] - self.distance
        y_square = self.points['y0'] - self.distance
        
        # Use array slicing instead of cv2.resize when possible
        if self.square_size == frame.shape[0]:
            return frame[y_square:y_square + self.square_size, 
                        x_square:x_square + self.square_size]
        else:
            cropped = frame[y_square:y_square + self.square_size, 
                           x_square:x_square + self.square_size]
            return cv2.resize(cropped, (self.square_size, self.square_size))
        
    def apply_masks(self, frame):
        """Apply binary and radius masks"""
        # Convert to grayscale if not already
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) > 2 else frame
        
        # Use in-place operations
        cv2.bitwise_and(gray, gray, dst=self.temp_roi, mask=cv2.bitwise_not(self.mask))
        cv2.bitwise_and(self.temp_roi, self.temp_roi, dst=gray, mask=self.mask2)
        
        # Return views instead of copies for ROIs
        left_ROI = gray[:, :self.mask_x1]
        right_ROI = gray[:, self.mask_x2:]
        
        return left_ROI, right_ROI
        
    def analyze_wings(self, rois):
        """Analyze left and right wing regions"""
        left_ROI, right_ROI = rois
        
        # Process left wing
        left_data = process_wing_region(
            roi=left_ROI,
            hinge_point=self.left_point_new,
            is_right_wing=False
        )
        
        # Process right wing
        right_data = process_wing_region(
            roi=right_ROI,
            hinge_point=self.right_point_new,
            is_right_wing=True,
            mask_offset=self.mask_x2
        )
        
        return left_data, right_data
        
    def calculate_angles(self, left_data, right_data):
        """Calculate final angles and return results"""
        left_angle, left_min_point, left_area = left_data
        right_angle, right_min_point, right_area = right_data
        
        # Validate angles
        min_angle = self.config['wingbeat']['min_angle_threshold']
        max_angle = self.config['wingbeat'].get('max_angle_threshold', 180)
        
        # Force angles to 0 if they're invalid
        if not (min_angle <= left_angle <= max_angle):
            left_angle = 0
            left_min_point = self.left_point_new
        
        if not (min_angle <= right_angle <= max_angle):
            right_angle = 0
            right_min_point = self.right_point_new
        
        # Calculate delta only if both angles are valid
        if left_angle == 0 or right_angle == 0:
            delta_angle_rl = 0
        else:
            delta_angle_rl = right_angle - left_angle
            
        return {
            'left_angle': left_angle,
            'right_angle': right_angle,
            'delta_angle': delta_angle_rl,
            'left_point': left_min_point,
            'right_point': right_min_point
        }
        
    def process_frame(self, frame):
        """Main frame processing method"""
        self.last_cropped_frame = self.crop_frame(frame)
        processed_frame = self.apply_masks(self.last_cropped_frame)
        left_data, right_data = self.analyze_wings(processed_frame)
        results = self.calculate_angles(left_data, right_data)
        
        # Draw only wing lines on the cropped frame
        if results['left_angle'] >= self.config['wingbeat']['min_angle_threshold'] and \
           results['right_angle'] >= self.config['wingbeat']['min_angle_threshold']:
            # Draw left wing line
            cv2.line(self.last_cropped_frame,
                    self.left_point_new,
                    results['left_point'],
                    self.config['wingbeat']['line_color'],
                    self.config['wingbeat']['line_thickness'])
            
            # Draw right wing line
            cv2.line(self.last_cropped_frame,
                    self.right_point_new,
                    results['right_point'],
                    self.config['wingbeat']['line_color'],
                    self.config['wingbeat']['line_thickness'])
        
        return results
        
    def get_last_cropped_frame(self):
        """Return the last cropped frame"""
        return self.last_cropped_frame

def process_wing_region(roi, hinge_point, is_right_wing=False, mask_offset=0):
    """Process wing region to calculate angle and area."""
    # Create mask for wing intensity range
    wing_mask = cv2.inRange(
        roi, 
        CONFIG['intensity_thresholds']['wing_min'],
        CONFIG['intensity_thresholds']['wing_max']
    )
    
    # Apply morphological operations
    kernel = np.ones(CONFIG['wingbeat']['kernel_size'], np.uint8)
    
    # First dilate to connect wing regions
    wing_mask = cv2.dilate(
        wing_mask, 
        kernel, 
        iterations=CONFIG['wingbeat']['dilation_iterations']
    )
    
    # Then erode to clean up noise
    wing_mask = cv2.erode(
        wing_mask, 
        kernel, 
        iterations=CONFIG['wingbeat']['erosion_iterations']
    )
    
    # Find contours in processed mask
    contours, _ = cv2.findContours(wing_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Early return if no contours found
    if not contours:
        return 0, hinge_point, 0
    
    # Get largest contour
    areas = np.array([cv2.contourArea(c) for c in contours])
    max_idx = np.argmax(areas)
    max_contour = contours[max_idx]
    max_area = areas[max_idx]
    
    # Early return if contour area is too small
    if max_area < CONFIG['wingbeat']['min_contour_area']:
        return 0, hinge_point, 0
    
    # Find highest point (minimum y)
    min_y = float('inf')
    wing_tip = None
    
    for point in max_contour[:, 0, :]:
        if point[1] < min_y:  # Lower y is higher in image
            min_y = point[1]
            # Apply offset for right wing
            x_coord = point[0] + mask_offset if is_right_wing else point[0]
            wing_tip = (x_coord, point[1])
            
    # Early return if no wing tip found
    if wing_tip is None:
        return 0, hinge_point, 0
    
    # Calculate angle
    delta_x = wing_tip[0] - hinge_point[0]
    delta_y = hinge_point[1] - wing_tip[1]
    
    # Check if the detected point is too close to hinge
    min_distance = CONFIG['wingbeat'].get('min_wing_length', 10)  # Add this to config if not present
    if (delta_x**2 + delta_y**2) < min_distance**2:
        return 0, hinge_point, 0
        
    if is_right_wing:
        angle = 180 - np.degrees(np.arctan2(delta_x, delta_y))
    else:
        angle = 180 - np.degrees(np.arctan2(-delta_x, delta_y))
    
    # Validate angle is in reasonable range
    if angle < 0 or angle > 180:
        return 0, hinge_point, 0
        
    return angle, wing_tip, max_area


def run_analysis(video_path: str, output_dir: Path, make_video: bool = False) -> None:
    """Run the complete wingbeat analysis pipeline."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for both video and data files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create analyzer first
    analyzer = WingbeatAnalyzer(CONFIG)
    
    # Setup handlers
    video_handler = VideoHandler(
        source_path=video_path,
        output_path=output_dir / f"output_{timestamp}.avi" if make_video else None,
        make_video=make_video
    )
    
    # Pass the same timestamp to DataManager
    data_manager = DataManager(
        output_dir / f"wingbeat_data_{timestamp}.csv"
    )
    
    try:
        # Get first frame for point selection
        first_frame = video_handler.read_frame()
        if first_frame is None:
            raise RuntimeError("Could not read first frame")
            
        # Setup points through UI
        analyzer.setup_points_ui(first_frame)
        
        # Now we can update the video writer with the correct frame size
        if make_video:
            video_handler.update_frame_size((analyzer.square_size, analyzer.square_size))
        
        # Main processing loop
        frame_count = 0
        
        while True:
            frame = video_handler.read_frame()
            if frame is None:
                break
                
            frame_count += 1
            logger.info(f"Processing frame: {frame_count}")
            
            # Process frame and get cropped version
            cropped_frame = analyzer.crop_frame(frame)
            results = analyzer.process_frame(frame)
            results['frame'] = frame_count
            
            # Save data
            data_manager.write_row(results)
            
            # Write video frame if enabled
            if make_video:
                # Draw wing lines on cropped frame
                if results['left_angle'] >= CONFIG['min_angle_threshold'] and \
                   results['right_angle'] >= CONFIG['min_angle_threshold']:
                    cv2.line(cropped_frame, 
                            analyzer.left_point_new, 
                            results['left_point'],
                            CONFIG['video_line_color'],
                            CONFIG['video_line_thickness'])
                    cv2.line(cropped_frame,
                            analyzer.right_point_new,
                            results['right_point'],
                            CONFIG['video_line_color'],
                            CONFIG['video_line_thickness'])
                
                video_handler.write_frame(cropped_frame)  # Write cropped frame instead of original
            
                
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise
        
    finally:
        # Cleanup
        video_handler.release()
        data_manager.close()
        cv2.destroyAllWindows()
        
        logger.info(f"Analysis complete. Processed {frame_count} frames")

def main():
    """Main entry point"""
    try:
        video_path = CONFIG['paths']['video_dir'] / "2024-12-06_115450_233.avi"
        output_dir = CONFIG['paths']['output_dir']
        
        run_analysis(
            video_path=video_path,
            output_dir=output_dir,
            make_video=True
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return 1
        
    return 0

if __name__ == '__main__':
    exit(main())

class AnalysisError(Exception):
    """Custom exception for analysis errors"""
    pass

def setup_logging():
    """Configure logging with proper formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )