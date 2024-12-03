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

def plot_summary_bar_chart(df):
    """Generates a bar chart comparing aggregate token usage for the top 10 models."""
    # Aggregate total tokens for each model and convert to millions
    model_totals = df.groupby('model')['total_tokens'].sum() / 1e6
    model_totals = model_totals.sort_values(ascending=False).head(10)

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    model_totals.plot(kind='bar', color='skyblue')
    
    plt.title('Top 10 Models by Total Token Usage (in Millions)')
    plt.xlabel('Model')
    plt.ylabel('Total Tokens (Millions)')
    plt.xticks(rotation=45, ha='right')  # Rotate labels and align them
    plt.grid(axis='y')

    # Adjust margins to prevent label cut-off
    plt.subplots_adjust(bottom=0.3)

    plt.savefig('top_10_models_token_usage.png')
    plt.close()
    print("Summary bar chart saved as top_10_models_token_usage.png")

def plot_daily_token_trends(df):
    """Plots daily trends for input, output, and total tokens summed across all models."""
    # Convert timestamp to datetime and extract date
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # Aggregate tokens by day and convert to millions
    daily_totals = df.groupby('date').sum(numeric_only=True)[
        ['prompt_tokens', 'completion_tokens', 'total_tokens']
    ] / 1e6

    # Plot the daily token trends
    plt.figure(figsize=(12, 6))
    plt.plot(daily_totals.index, daily_totals['prompt_tokens'], label='Prompt Tokens', marker='o')
    plt.plot(daily_totals.index, daily_totals['completion_tokens'], label='Completion Tokens', marker='x')
    plt.plot(daily_totals.index, daily_totals['total_tokens'], label='Total Tokens', marker='s')

    plt.title('Daily Token Trends (Summed Across All Models, in Millions)')
    plt.xlabel('Date')
    plt.ylabel('Tokens (Millions)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45, ha='right')

    # Adjust margins to prevent label cut-off
    plt.subplots_adjust(bottom=0.25)

    plt.savefig('daily_token_trends.png')
    plt.close()
    print("Daily token trends plot saved as daily_token_trends.png")

def main(jsonl_file, excel_output):
    # Step 1: Load the JSON-L data into a DataFrame
    df = read_jsonl_to_dataframe(jsonl_file)

    # Step 2: Save the data to an Excel spreadsheet
    save_to_excel(df, excel_output)

    # Step 3: Generate the top 10 models summary bar chart
    plot_summary_bar_chart(df)

    # Step 4: Generate the daily token trends plot
    plot_daily_token_trends(df)

# Usage example
if __name__ == "__main__":
    jsonl_file = 'requests.log'  # Replace with your JSON-L file path
    excel_output = 'summary.xlsx'  # Replace with desired Excel output path
    main(jsonl_file, excel_output)

