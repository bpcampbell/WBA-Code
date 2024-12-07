from vidgear.gears import WriteGear
import cv2
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class VideoHandler:
    def __init__(self, source_path, output_path=None, make_video=False, frame_size=None):
        """Initialize video capture and writer.
        
        Args:
            source_path: Path to source video file
            output_path: Path to output video file (if recording)
            make_video: Whether to record output video
            frame_size: Tuple of (width, height) for output video
        """
        self.source_path = Path(source_path)
        self.output_path = Path(output_path) if output_path else None
        self.make_video = make_video
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(str(self.source_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source_path}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Use provided frame size for output if specified
        if frame_size:
            self.output_width, self.output_height = frame_size
        else:
            self.output_width, self.output_height = self.width, self.height
        
        # Initialize video writer if needed
        self.writer = None
        if self.make_video:
            self.setup_video_writer()
            
    def setup_video_writer(self):
        """Initialize WriteGear writer with optimized settings"""
        if not self.output_path:
            raise ValueError("Output path not specified for video recording")
            
        # Define output parameters for WriteGear
        output_params = {
            "-input_framerate": self.fps,
            "-c:v": "libx264",  # Use H.264 codec
            "-crf": 17,  # Constant Rate Factor (0-51, lower means better quality)
            "-preset": "fast",  # Encoding speed preset
            "-tune": "zerolatency",  # Optimize for low-latency streaming
            "-color_range": 2,  # Full color range
            "-pix_fmt": "yuv420p"  # Pixel format for better compatibility
        }
        
        self.writer = WriteGear(
            output=str(self.output_path),
            compression_mode=True,
            logging=False,  # Changed from True to False to disable logging
            **output_params
        )
        
    def read_frame(self):
        """Read and return a frame from the video source."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
        
    def write_frame(self, frame):
        """Write frame using WriteGear if enabled."""
        if self.make_video and self.writer:
            self.writer.write(frame)
            
    def release(self):
        """Release video capture and writer resources"""
        self.cap.release()
        if self.writer:
            self.writer.close()  # WriteGear uses close() instead of release()
            
    def update_frame_size(self, frame_size):
        """Update the output frame size and reinitialize writer if needed."""
        self.output_width, self.output_height = frame_size
        if self.make_video:
            if self.writer is not None:
                self.writer.close()
            self.setup_video_writer()