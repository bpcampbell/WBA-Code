from pathlib import Path
import logging, cv2
import pandas as pd
from src.experiment_manager import ExperimentManager
from src.config import CONFIG


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the real-time wingbeat analyzer and optic flow experiment"""
    try:
        # Use test video from root directory
        root_dir = Path(__file__).parent
        video_path = root_dir / "test_video.avi"
        output_dir = root_dir / "output"
        
        # Ensure video exists
        if not video_path.exists():
            raise FileNotFoundError(f"Test video not found: {video_path}")
        
        # Get parameters from config
        experiment_config = CONFIG['experiment']
        frame_size = experiment_config['frame_size']
        gain_cycle = [(cycle['duration'], cycle['gain']) 
                      for cycle in experiment_config['gain_cycle']]
        
        logger.info("Initializing experiment...")
        experiment = ExperimentManager(
            frame_size=frame_size,
            video_path=video_path,
            min_amplitude=experiment_config['min_amplitude'],
            max_amplitude=experiment_config['max_amplitude']
        )
        
        # Run experiment and collect data
        result = experiment.run_experiment(experiment_config['name'], gain_cycle)
        if result is None or result[0] is None:  # Check both data and timestamp
            logger.info("Experiment cancelled or failed")
            return 1
            
        data, timestamp = result
        
        # Save data
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"experiment_data_{timestamp}.csv"
        
        df = pd.DataFrame(data, columns=[
           'time',
            'left_angle',
            'right_angle', 
            'delta_angle',
            'frame',
            'phase',
            'speed',
            'gain'
        ])
        df.to_csv(output_file, index=False)
        
        logger.info(f"Data saved to: {output_file}")
        logger.info("Experiment completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        return 1
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    exit(main()) 
