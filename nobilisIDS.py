import subprocess
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import joblib
import time
import csv
import os
import signal
import sys

MODEL_PATHS = {
    "DDoS": "models/ddos.pkl",
    "Infiltration": "models/infiltration.pkl",
    "Portscan": "models/portscan.pkl",
    "Webattacks": "models/webattacks.pkl"
}

# Global variables to hold the subprocesses
ids_process = None
cicflowmeter_process = None


def run_cicflowmeter(interface):
    global cicflowmeter_process
    command = ["sudo", "cicflowmeter", "-i", interface, "-c"]

    # Run the command and store the process object
    cicflowmeter_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_model(model_file):
    # Load the pre-trained model
    print("Loading the trained model...")
    model = joblib.load(model_file)
    return model


def choose_model():
    # Ask the user to choose a model
    print("Available models:")
    for idx, model_name in enumerate(MODEL_PATHS.keys(), start=1):
        print(f"{idx}. {model_name}")

    while True:
        try:
            choice = int(input("Enter the number of the model you want to use: "))
            if 1 <= choice <= len(MODEL_PATHS):
                break
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid choice. Please enter a valid number.")

    model_name = list(MODEL_PATHS.keys())[choice - 1]
    model_file = MODEL_PATHS[model_name]
    model = load_model(model_file)

    print(f"Loaded model: {model_name}")
    return model


def predict_flow(model, flow):
    # Reshape the flow array to 2D
    flow = flow.reshape(1, -1)

    # Perform prediction on the flow
    prediction = model.predict(flow)
    return prediction[0]  # Return the first element of the prediction array as a string


def get_total_lines(csv_file):
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        total_lines = sum(1 for _ in reader)
    return total_lines


def preprocess_flow(line):
    # Create a DataFrame with the line
    df = pd.DataFrame([line])

    # Drop missing values
    df.dropna(inplace=True)

    # Convert column names to strings and remove whitespaces
    df.columns = df.columns.astype(str).str.strip()

    # Check for infinity values and replace with np.nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Reshape the DataFrame to a 2D array
    X = df.values.reshape(1, -1)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    return X


def stop_subprocesses():
    global ids_process, cicflowmeter_process

    # Terminate the ids subprocess if running
    if ids_process and ids_process.poll() is None:
        ids_process.terminate()

    # Terminate the cicflowmeter subprocess if running
    if cicflowmeter_process and cicflowmeter_process.poll() is None:
        cicflowmeter_process.terminate()

    sys.exit(0)

def ask_delete_existing_file(csv_file):
    while True:
        choice = input(f"CSV file '{csv_file}' already exists. Do you want to delete it? (y/n): ")
        if choice.lower() == 'y':
            os.remove(csv_file)
            break
        elif choice.lower() == 'n':
            break
        else:
            print("Invalid choice. Please enter 'y' or 'n'.")


def main():
    global ids_process
    model = choose_model()

    # Ask the user for the interface to use
    interface = input("Enter the interface name: ")
    csv_file = f"{interface}.csv"

    # Ask the user to delete existing .csv file or not
    ask_delete_existing_file(csv_file)

    # Start cicflowmeter in parallel
    run_cicflowmeter(interface)

    # Set up a signal handler to catch Ctrl+C
    signal.signal(signal.SIGINT, lambda signum, frame: stop_subprocesses())

    # Wait until the CSV file is created
    while not os.path.exists(csv_file):
        print("Flow dump is not created, waiting...")
        time.sleep(15)

    last_line_number = 1

    # Monitor the CSV file for new flows
    while True:
        # Get the current number of lines in the CSV file
        current_lines = get_total_lines(csv_file)

        # Check if there are any new lines
        if current_lines > last_line_number:
            new_lines = current_lines - last_line_number

            # Process each new line (flow)
            with open(csv_file, "r") as file:
                reader = csv.reader(file)
                lines = list(reader)
                for i in range(new_lines):
                    line_number = last_line_number + i + 1

                    # Read the line from the CSV file if it exists
                    if len(lines) >= line_number:
                        line = lines[line_number - 1]

                        # Preprocess the line along with the first line
                        flow = preprocess_flow(line)

                        if flow is not None:
                            # Use the loaded model to predict if the flow is a cyberattack
                            prediction = predict_flow(model, flow)
                            if prediction == 'BENIGN':
                                # Print normal activity
                                print(f"Flow №{line_number}: Prediction: {prediction}", end="\r")
                            else:
                                print(f"\nFlow №{line_number}: Prediction: {prediction}")
                                print(f"Possible intrusion in flow №{line_number}, please take actions!")

            # Update the last processed line number
            last_line_number = current_lines

        # Wait for a short time before checking for new lines again
        time.sleep(1)


if __name__ == "__main__":
    main()
