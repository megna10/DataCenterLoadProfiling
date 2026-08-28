import pandas as pd

def _prepare_dataframe(data):

    # prevents modifying existing data
    df = data.copy()

    # formats timestamp to be read as time instead of a string
    if not pd.api.types.is_datetime64_any_dtype(df['TIMESTAMP']):
        # Parses the string timestamps into actual Pandas/NumPy datetime objects, which are necessary for time-based manipulation
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='ISO8601')
    return df.set_index('TIMESTAMP')

def generate_request_arrival_rate(data):
    # ensure timestamps are parsed and set as the index
    df = _prepare_dataframe(data)

    # 1. Groups the data into fixed 1-second intervals (bins) based on the datetime index
    # 2. Counts the total number of rows (requests) that fall into each 1-second bin
    # 3. Replaces NaN values with 0 for any 1-second interval that had zero requests
    return df.resample('1s').size().fillna(0).rename('request_arrival_rate')


def generate_input_token_rate(data):
    df = _prepare_dataframe(data)
    return (
        df['ContextTokens'].resample('1s').sum().fillna(0).rename('input_token_rate')
    )


def generate_output_token_rate(data):
    df = _prepare_dataframe(data)
    return (
        df['GeneratedTokens']
        .resample('1s')
        .sum()
        .fillna(0)
        .rename('output_token_rate')
    )

def main():
    """Main execution block where workflow functions are called."""
    dataset = pd.read_csv('llm_tokens.csv')

    request_rates = generate_request_arrival_rate(dataset)
    input_rates = generate_input_token_rate(dataset)
    output_rates = generate_output_token_rate(dataset)

    metrics = pd.concat([request_rates, input_rates, output_rates], axis=1).fillna(0)
    print(metrics)

# --- RUN SCRIPT ---
if __name__ == "__main__":
    main()
