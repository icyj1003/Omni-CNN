import os
import csv
from tensorflow.python.summary.summary_iterator import summary_iterator

def save_tensorboard_to_csv(tfevents_file, output_csv_file):
    """
    Extracts data from a TensorBoard log file and saves it to a CSV file.

    Args:
        tfevents_file (str): Path to the TensorBoard log file (e.g., 'runs/.../events.out.tfevents').
        output_csv_file (str): Path to the output CSV file.
    """
    # Check if the TensorBoard file exists
    if not os.path.exists(tfevents_file):
        raise FileNotFoundError(f"File not found: {tfevents_file}")

    # Open the CSV file for writing
    with open(output_csv_file, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write the header
        writer.writerow(["Tag", "Step", "Value"])

        # Read the TensorBoard log file
        for event in summary_iterator(tfevents_file):
            for value in event.summary.value:
                if value.HasField("simple_value"):  # Filter for scalar values
                    writer.writerow([value.tag, event.step, value.simple_value])

    print(f"Data saved to {output_csv_file}")
    
tfevents_path = "runs/Dec04_12-42-55_anda-Z690-AORUS-XTREME/events.out.tfevents.1733283775.anda-Z690-AORUS-XTREME.3625946.0"
output_csv_path = "results/tensorboard_data_Dec04_12-42-55.csv"

save_tensorboard_to_csv(tfevents_path, output_csv_path)