import pandas as pd
import matplotlib.pyplot as plt

def plot_wingbeat_amplitude_and_speed():
    file_name = "test1_data.csv"

    try:
        df = pd.read_csv(file_name)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Wingbeat Amplitude', color=color)
        ax1.plot(df['Time'], df['Wingbeat Amplitude'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a Second Axes that shares the same x-axis

        color = 'tab:red'
        ax2.set_ylabel('Optic Flow Speed', color=color)  # we already handled the x-label with ax1
        ax2.plot(df['Time'], df['Speed'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

    except FileNotFoundError:
        print(f"File '{file_name}' not found. Please check the file name and path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    plot_wingbeat_amplitude_and_speed()