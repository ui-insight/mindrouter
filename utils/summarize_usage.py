import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def read_jsonl_to_dataframe(file_path):
    """Reads a JSON-L file and loads it into a Pandas DataFrame."""
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

def save_to_excel(df, output_path):
    """Saves the DataFrame to an Excel file."""
    df.to_excel(output_path, index=False)
    print(f"Spreadsheet saved to {output_path}")

def plot_token_utilization(df):
    """Generates matplotlib visualizations for token utilization over time."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Group data by model and plot token utilization over time
    models = df['model'].unique()
    
    for model in models:
        model_data = df[df['model'] == model]
        
        plt.figure()
        plt.plot(model_data['timestamp'], model_data['prompt_tokens'], label='Prompt Tokens', marker='o')
        plt.plot(model_data['timestamp'], model_data['completion_tokens'], label='Completion Tokens', marker='x')
        plt.plot(model_data['timestamp'], model_data['total_tokens'], label='Total Tokens', marker='s')
        
        plt.title(f'Token Utilization over Time - {model}')
        plt.xlabel('Timestamp')
        plt.ylabel('Tokens')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        plt.savefig(f"{model}_token_utilization.png")
        plt.close()
        print(f"Plot saved for model: {model}")

def main(jsonl_file, excel_output):
    # Step 1: Load the JSON-L data into a DataFrame
    df = read_jsonl_to_dataframe(jsonl_file)

    # Step 2: Save the data to an Excel spreadsheet
    save_to_excel(df, excel_output)

    # Step 3: Generate token utilization plots
    plot_token_utilization(df)

# Usage example
if __name__ == "__main__":
    jsonl_file = 'requests.log'  # Replace with your JSON-L file path
    excel_output = 'summary.xlsx'  # Replace with desired Excel output path
    main(jsonl_file, excel_output)

