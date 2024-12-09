import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_latest_csv():
    """Get the most recent CSV file from the output directory."""
    output_dir = Path("output")
    csv_files = list(output_dir.glob("experiment_data_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("No experiment data files found in output directory")
        
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading data from: {latest_file}")
    return latest_file

def create_plot():
    """Create and display interactive plot of wingbeat data."""
    try:
        csv_file = get_latest_csv()
        df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Create figure with secondary y-axis for gain
    fig = go.Figure()
    
    # Plot wingbeat angles
    fig.add_trace(go.Scatter(
        x=df['time'], 
        y=df['left_angle'],
        name='Left Wing',
        line=dict(color='#00ff00', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['time'], 
        y=df['right_angle'],
        name='Right Wing',
        line=dict(color='#ff0000', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['time'], 
        y=df['delta_angle'],
        name='Wing Amplitude',
        line=dict(color='#ffffff', width=2)
    ))
    
    # Add gain as a step plot with secondary y-axis
    fig.add_trace(go.Scatter(
        x=df['time'], 
        y=df['gain'],
        name='Gain',
        line=dict(shape='hv', color='#ff00ff', width=2),
        yaxis='y2'
    ))
    
    # Update layout with secondary y-axis
    fig.update_layout(
        title='Wingbeat Analysis',
        xaxis=dict(
            title='Time (ms)',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            title='Angle (degrees)',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis2=dict(
            title='Gain',
            overlaying='y',
            side='right',
            showgrid=False,
            range=[0, max(df['gain']) * 1.1]  # Give some headroom
        ),
        template='plotly_dark',
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1,
            x=0.01,
            y=0.99,
            xanchor='left',
            yanchor='top'
        )
    )
    
    # Show the plot
    fig.show()

if __name__ == "__main__":
    create_plot() 