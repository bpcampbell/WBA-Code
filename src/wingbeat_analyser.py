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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WingbeatAnalyzer:
    def __init__(self, config):
        self.config = config
        self.points = {'x0': None, 'y0': None, 'x1': None, 'y1': None,
                      'x2': None, 'y2': None, 'x3': None, 'y3': None,
                      'x4': None, 'y4': None}
        self.click_count = 0
        self.setup_masks()
        self.setup_points()
        
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
        central_point = (self.points['x0'], self.points['y0'])
        region_point = (self.points['x2'], self.points['y2'])
        left_point = (self.points['x3'], self.points['y3'])
        right_point = (self.points['x4'], self.points['y4'])
        
        # Calculate distance and square region
        self.distance = int(np.sqrt((region_point[0] - central_point[0]) ** 2 + 
                                  (region_point[1] - central_point[1]) ** 2))
        self.square_size = 2 * self.distance
        
        # Calculate new points
        self._calculate_transformed_points(central_point, left_point, right_point)
        
        # Initialize masks
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
        # Reuse pre-allocated array for binary threshold
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, self.binary_frame)
        cv2.threshold(self.binary_frame, self.config['threshold'], 255, cv2.THRESH_BINARY, self.binary_frame)
        
        # Use in-place operations
        cv2.bitwise_and(self.binary_frame, self.binary_frame, dst=self.temp_roi, mask=cv2.bitwise_not(self.mask))
        cv2.bitwise_and(self.temp_roi, self.temp_roi, dst=self.binary_frame, mask=self.mask2)
        
        # Apply erosion and dilation in-place
        kernel = np.ones(self.config['kernel_size'], np.uint8)
        cv2.dilate(self.binary_frame, kernel, dst=self.temp_roi, 
                   iterations=self.config['dilation_iterations'])
        cv2.erode(self.temp_roi, kernel, dst=self.binary_frame, 
                  iterations=self.config['erosion_iterations'])
        
        # Return views instead of copies for ROIs
        left_ROI = self.binary_frame[:, :self.mask_x1]
        right_ROI = self.binary_frame[:, self.mask_x2:]
        
        return left_ROI, right_ROI
        
    def analyze_wings(self, rois):
        """Analyze left and right wing regions"""
        left_ROI, right_ROI = rois
        left_data = process_left_region(
            left_ROI=left_ROI,
            central_point_new=self.central_point_new,
            left_point_new=self.left_point_new
        )
        right_data = process_right_region(
            right_ROI=right_ROI,
            central_point_new=self.central_point_new,
            right_point_new=self.right_point_new,
            mask_x2=self.mask_x2
        )
        return left_data, right_data
        
    def calculate_angles(self, left_data, right_data):
        """Calculate final angles and return results"""
        left_angle, left_min_point, left_area = left_data
        right_angle, right_min_point, right_area = right_data
        
        if left_angle < self.config['min_angle_threshold'] or right_angle < self.config['min_angle_threshold']:
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
        cropped_frame = self.crop_frame(frame)
        processed_frame = self.apply_masks(cropped_frame)
        left_data, right_data = self.analyze_wings(processed_frame)
        return self.calculate_angles(left_data, right_data)
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get the point number based on click count
            point_num = self.click_count
            if point_num < 5:  # We need 5 points total
                # Set both x and y coordinates at once
                self.points[f'x{point_num}'] = x
                self.points[f'y{point_num}'] = y
                logger.info(f"Point {point_num} set to: ({x}, {y})")
                self.click_count += 1

    def setup_points_ui(self, frame):
        """Setup UI for point selection"""
        logger.info("Please select 5 points in order:")
        logger.info("1. Centre of the body")
        logger.info("2. Centre of the head")
        logger.info("3. Point to track on the wing")
        logger.info("4. Left wing hinge")
        logger.info("5. Right wing hinge")
        
        cv2.namedWindow("FrameClick")
        cv2.setMouseCallback("FrameClick", self.mouse_callback)
        
        # Keep showing frame until all points are selected
        while self.click_count < 5:
            cv2.imshow("FrameClick", frame)
            cv2.pollKey()
        
        # Immediately destroy window after last point
        cv2.destroyWindow("FrameClick")
        cv2.waitKey(1)  # Force window destruction
        
        self.initialize_processing(frame)
        logger.info("Point selection and initialization complete")


def process_left_region(left_ROI: np.ndarray, 
                       central_point_new: Tuple[int, int], 
                       left_point_new: Tuple[int, int]) -> Tuple[float, Tuple[int, int], float]:
    """Process the left wing region to calculate angle and area."""
    contours, _ = cv2.findContours(left_ROI, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, central_point_new, 0
        
    # Use array operations instead of max()
    areas = np.array([cv2.contourArea(c) for c in contours])
    max_idx = np.argmax(areas)
    max_contour = contours[max_idx]
    
    # Draw contour directly to ROI
    cv2.drawContours(left_ROI, [max_contour], -1, 255, thickness=cv2.FILLED)
    
    # Use numpy operations for point finding
    points = np.where(left_ROI == 255)
    min_y_idx = np.argmin(points[0])
    left_min_point = (points[1][min_y_idx], points[0][min_y_idx])
    
    # Vectorized angle calculation
    delta = np.array([left_point_new[0] - left_min_point[0],
                     left_point_new[1] - left_min_point[1]])
    left_angle = 180 - np.degrees(np.arctan2(delta[0], delta[1]))
    
    return left_angle, left_min_point, areas[max_idx]


def process_right_region(right_ROI: np.ndarray, 
                        central_point_new: Tuple[int, int], 
                        right_point_new: Tuple[int, int], 
                        mask_x2: int) -> Tuple[float, Tuple[int, int], float]:
    """Process the right wing region to calculate angle and area."""
    right_contours, _ = cv2.findContours(right_ROI, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(right_contours) == 0:
        right_angle = 0
        right_min_point = central_point_new
        right_area = 0
    else:
        right_max_contour = max(right_contours, key=cv2.contourArea)
        right_area = cv2.contourArea(right_max_contour)
        new_right_ROI = np.zeros_like(right_ROI)
        cv2.drawContours(new_right_ROI, [right_max_contour], -1, 255, thickness=cv2.FILLED)

        right_y, right_x = np.where(new_right_ROI == 255)
        right_min_y_index = np.argmin(right_y)
        right_min_y_coord = (right_x[right_min_y_index], right_y[right_min_y_index])
        right_min_point = (right_min_y_coord[0] + mask_x2, right_min_y_coord[1])
        right_delta_x = right_min_point[0] - right_point_new[0]
        right_delta_y = right_point_new[1] - right_min_point[1]
        right_angle = 180 - np.degrees(np.arctan2(right_delta_x, right_delta_y))

    return right_angle, right_min_point, right_area


def process_frame(frame, left_ROI, right_ROI):
    """Process a single frame to get wing angles and points."""
    # Process left region
    left_angle, left_min_point, left_area = process_left_region(frame)
    
    # Process right region
    right_angle, right_min_point, right_area = process_right_region(frame)
    
    # Calculate wing beat amplitude difference (dWBA)
    if left_angle < 50 or right_angle < 50:
        delta_angle_rl = 0
    else:
        delta_angle_rl = right_angle - left_angle
        
    return left_angle, right_angle, delta_angle_rl, left_min_point, right_min_point


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
            
            # Check for quit
            if cv2.pollKey() & 0xFF == ord('q'):
                logger.info("User requested stop")
                break
                
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