import pandas as pd
import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------------------------
# SERVER CONFIGURATION
# ---------------------------------------------------------------------------
# Defines the electrical and performance characteristics of each server type.
#
# P_idle:
#     Power consumed by one server when it is idle, in watts.
#
# P_peak:
#     Maximum IT power consumed by one server, in watts.
#
# gpus_per_server:
#     Number of GPUs contained in one server.
#
# R_prefill:
#     Estimated token-processing rate for the prompt/input (prefill) phase,
#     measured in tokens/second.
#
# R_decode:
#     Estimated token-processing rate for the generated/output (decode) phase,
#     measured in tokens/second.
#
# "inf" for R_decode means that decode time is intentionally treated as zero.
# ---------------------------------------------------------------------------

SERVER_CONFIGS = {

        "Standard": {'P_idle': 579.0, 'P_peak': 2936.0, 'gpus_per_server': 4, "R_prefill": 152.05, "R_decode": float("inf")},
        'Dense': {'P_idle': 1036.0, 'P_peak': 5768.0, 'gpus_per_server': 4, "R_prefill": 987 , "R_decode": float("inf")},
        'Extreme': {'P_idle': 1747.0, 'P_peak': 11011.0, 'gpus_per_server': 8, "R_prefill": 1792.26, "R_decode": float("inf")},
    }

def _prepare_dataframe(data):
    """
    Create a copy of the input dataset and convert timestamps to datetime.

    Parameters
    ----------
    data : pandas.DataFrame
        Input workload dataset. It must contain a "TIMESTAMP" column.

    Returns
    -------
    pandas.DataFrame
        A copy of the input dataframe with "TIMESTAMP" converted to
        pandas datetime objects.

    """
    df = data.copy()

    # 2024-01-01T12:30:00
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="ISO8601")
    return df

# ---------------------------------------------------------------------------
# DATASET SPLITTING
# ---------------------------------------------------------------------------

def decrease_dataset():
    """
    Split the full LLM inference trace into seven separate daily CSV files.

    The input dataset is expected to contain at least:
        - TIMESTAMP
        - ContextTokens
        - GeneratedTokens

    The function finds the earliest timestamp in the dataset and then creates
    one CSV file for each of the following seven 24-hour periods.
    """

    INPUT_FILE = "llm_inference_files/llm_tokens_conv.csv"

    # Only load the columns you actually need
    df = pd.read_csv(
        INPUT_FILE,
        usecols=[
            "TIMESTAMP",
            "ContextTokens",
            "GeneratedTokens"
        ]
    )

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="ISO8601")

    # Use the earliest timestamp as the beginning of Day 1.
    start_time = df["TIMESTAMP"].min()

    for day in range(7):

        # Calculate the beginning and end of the current day.
        day_start = start_time + pd.Timedelta(days=day)
        day_end = day_start + pd.Timedelta(days=1)

        day_df = df[
            (df["TIMESTAMP"] >= day_start) &
            (df["TIMESTAMP"] < day_end)
        ]

        filename = f"llm_inference_files/AzureLLMInferenceTrace_day{day + 1}.csv"

        day_df.to_csv(filename, index=False)

        print(
            f"Day {day + 1}: "
            f"{len(day_df):,} rows → {filename}"
        )

# ---------------------------------------------------------------------------
# R_prefill * R_decode
# ---------------------------------------------------------------------------

def generate_nnls_L40s():
    """
    Estimate gpu's prefill and decode throughput using benchmark data.

    The model assumes that total request processing time can be approximated
    as:
        Time = InputTokens / R_prefill
             + OutputTokens / R_decode

    The coefficients are estimated using Non-Negative Least Squares (NNLS).
    """
    # [Input_Tokens, Output_Tokens, Throughput_total (tokens/sec)]
    benchmark_data = np.array([
    [128,    128,   1523.52],
    [128,   2048,   1942.66],
    [128,   4096,   1440.23],
    [500,   2000,   1634.72],
    [1000,  1000,   1209.25],
    [2048,   128,    177.72],
    [2048,  2048,    969.68],
    [5000,   500,    249.52],
    [20000, 2000,    162.25]
])


     # Extract the input-token, output-token, and throughput columns.
    I = benchmark_data[:, 0]
    O = benchmark_data[:, 1]
    TP_total = benchmark_data[:, 2]

    # Construct Matrix A and Vector b (Ax = b)
    A = np.column_stack((I, O))
    b = (I + O) / TP_total #tokens/sec

    # Solve the linear system
    coeffs, _ = nnls(A, b)

    c_1 = coeffs[0]
    c_2 = coeffs[1]

    # Take reciprocals to get R_prefill and R_decode
    R_prefill = 1.0 / c_1
    R_decode  = 1.0 / c_2

    print(f"R_prefill = {R_prefill:.2f} tokens/sec")
    print(f"R_decode  = {R_decode:.2f} tokens/sec")

def generate_nnls_H100s():
    # [Input_Tokens, Output_Tokens, Throughput_total (tokens/sec)]
    benchmark_data = np.array([
        [128,   128,  11127.53],
        [128,  2048,  11511.93],
        [128, 4096, 426.32],
        [500,  2000,   9836.70],
        [1000, 1000,   7430.99],
        [2048,  128,   1302.60],
        [2048, 2048,   5480.03],
        [5000,  500,   1602.78],
        [20000, 2000,   920.19]
    ])

    # Extract columns
    I = benchmark_data[:, 0]
    O = benchmark_data[:, 1]
    TP_total = benchmark_data[:, 2]

    # Construct Matrix A and Vector b (Ax = b)
    A = np.column_stack((I, O))
    b = (I + O) / TP_total #tokens/sec

    # Solve the linear system
    coeffs, _ = nnls(A, b)

    c_1 = coeffs[0]
    c_2 = coeffs[1]

    # Take reciprocals to get R_prefill and R_decode
    R_prefill = 1.0 / c_1
    R_decode  = 1.0 / c_2

    print(f"R_prefill = {R_prefill:.2f} tokens/sec")
    print(f"R_decode  = {R_decode:.2f} tokens/sec")

def generate_nnls_H200s():
    # [Input_Tokens, Output_Tokens, Throughput_total (tokens/sec)]
    benchmark_data = np.array([
    [128,    128,   15355.84],
    [128,   2048,   21195.88],
    [128,   4096,   10574.06],
    [500,   2000,   17278.40],
    [1000,  1000,   13181.24],
    [2048,   128,    1983.03],
    [2048,  2048,   11142.47],
    [5000,   500,    2717.83],
    [20000, 2000,    1920.45]
])


    # Extract columns
    I = benchmark_data[:, 0]
    O = benchmark_data[:, 1]
    TP_total = benchmark_data[:, 2]

    # Construct Matrix A and Vector b (Ax = b)
    A = np.column_stack((I, O))
    b = (I + O) / TP_total #tokens/sec

    # Solve the linear system
    coeffs, _ = nnls(A, b)

    c_1 = coeffs[0]
    c_2 = coeffs[1]

    # Take reciprocals to get R_prefill and R_decode
    R_prefill = 1.0 / c_1
    R_decode  = 1.0 / c_2

    print(f"R_prefill = {R_prefill:.2f} tokens/sec")
    print(f"R_decode  = {R_decode:.2f} tokens/sec")

def calculate_duration(data, selected_deployment):
    """
    Calculate how long each LLM request occupies the selected server.

    Each request has two conceptual processing stages:

        1. Prefill:
           Processing the input/context tokens.

        2. Decode:
           Generating the output tokens.

    Total request duration is: prefill time + decode time

    Parameters
    ----------
    data : pandas.DataFrame
        LLM request trace containing:
        "TIMESTAMP", "ContextTokens", and "GeneratedTokens".

    selected_deployment : str
        Name of the server configuration in SERVER_CONFIGS.

    Returns
    -------
    pandas.DataFrame
        Original request data plus start/end timestamps and processing
        durations.
    """
    # gets r_prefill and r_decode for specific gpu
    spec = SERVER_CONFIGS[selected_deployment]

    df = _prepare_dataframe(data)
    df['start_time'] = df['TIMESTAMP']

    # Compute durations (Output / inf = 0.0)
    df['prefill_time_sec'] = df['ContextTokens'] / spec['R_prefill']
    df['decode_time_sec']  = df['GeneratedTokens'] / spec['R_decode']
    df['duration_sec']     = df['prefill_time_sec'] + df['decode_time_sec']

    df['end_time'] = df['start_time'] + pd.to_timedelta(df['duration_sec'], unit='s')

    return df

# ---------------------------------------------------------------------------
# WORKLOAD AGGREGATION
# ---------------------------------------------------------------------------

def aggregate_workload(processed_df, bin_size_minutes=5, chunk_size=500_000):
    """
    Aggregates request intervals into fixed-size workload bins.

    Designed for very large datasets (e.g. 27+ million rows)
    with limited RAM.

    Each request contributes the number of seconds that it is
    active inside each 5-minute bin.

    Example:
        Request: 10:02 -> 10:07

        10:00-10:05 = 180 seconds
        10:05-10:10 = 120 seconds

    Parameters
    ----------
    processed_df : pandas.DataFrame containing "start_time" and "end_time".

    bin_size_minutes : int, Width of each workload bin in minutes.

    chunk_size : int, Number of requests processed at once.

    Returns
    -------
    pandas.DataFrame
        One row per time bin containing:
            bin_start
            bin_end
            work_seconds
    """
    bin_seconds = bin_size_minutes * 60

    # calculates beginning and end of the timeline
    min_time = processed_df["start_time"].min().floor(f"{bin_size_minutes}min")

    max_time = processed_df["end_time"].max().ceil(f"{bin_size_minutes}min")

    # creates all five minute boundaries
    bin_edges = pd.date_range(
        start=min_time,
        end=max_time,
        freq=f"{bin_size_minutes}min"
    )

    # Everything except the last item. vice versa
    bin_starts = bin_edges[:-1]
    bin_ends = bin_edges[1:]

    num_bins = len(bin_starts)

    # converts timeline start into unix seconds
    timeline_start = min_time.value // 10**9

    # creates an array containing one 0 for every 5 min bin
    # eventually 0's become total seconds in a bin
    work_seconds = np.zeros(num_bins, dtype=np.float64)

    total_rows = len(processed_df)

    print(
        f"Processing {total_rows:,} rows "
        f"in chunks of {chunk_size:,}..."
    )

    # start at row 0 and move forward 500,000 rows at a time
    for chunk_start in range(0, total_rows, chunk_size):

        # usually chunk end is 500000 but on the final chunk it may be less than that
        chunk_end = min(chunk_start + chunk_size, total_rows)

        print(
            f"  Processing rows "
            f"{chunk_start:,} - {chunk_end:,} "
            f"({chunk_end / total_rows:.1%})"
        )

        # extract that chunked portion from the dataset
        chunk = processed_df.iloc[chunk_start:chunk_end]

        # Convert timestamps to Unix seconds
        request_starts = (chunk["start_time"].values.astype("datetime64[s]").astype(np.int64))

        request_ends = (chunk["end_time"].values.astype("datetime64[s]").astype(np.int64))

        # Remove invalid timestamps (NaT)
        invalid_timestamp = np.iinfo(np.int64).min

        # 1. start time isn't missing, 2. end time isn't missing
        valid_times = (
            (request_starts != invalid_timestamp)
            & (request_ends != invalid_timestamp)
            & (request_ends > request_starts)
        )

        # keep only valid requests
        request_starts = request_starts[valid_times]
        request_ends = request_ends[valid_times]

        # if chunk contains zero valid requests, then continue to process the next chunk
        if len(request_starts) == 0:
            continue

        # Determine starting and ending bins
        start_bins = ((request_starts - timeline_start) // bin_seconds).astype(np.int64)

        end_bins = ((request_ends - timeline_start)// bin_seconds).astype(np.int64)

        # bool for whether start/bin are same
        same_bin = start_bins == end_bins

        # ensures start and end are in the same bin and that both bins are in our timeline
        valid_same = (
            same_bin
            & (start_bins >= 0)
            & (start_bins < num_bins)
        )

        if np.any(valid_same):

            # for each same bin request, caluclate its duration and add it to that bin
            np.add.at(work_seconds, start_bins[valid_same], request_ends[valid_same] - request_starts[valid_same])


        multi_bin = (
            (start_bins != end_bins)
            & (start_bins >= 0)
            & (start_bins < num_bins)
        )

        if not np.any(multi_bin):
            continue


        s = request_starts[multi_bin] #start timestamp
        e = request_ends[multi_bin] # end timestamp
        sb = start_bins[multi_bin] # starting bin
        eb = end_bins[multi_bin] # ending bin

        # if request is 10:02 -> 10:17, first bin end is 10:05
        first_bin_end = (timeline_start + (sb + 1) * bin_seconds)

        # calculates how much time belonds in the first bin (180 seconds)
        first_bin_seconds = (first_bin_end - s)

        # add those seconds to the first bin
        np.add.at(work_seconds, sb, first_bin_seconds)

        # prevents writing outside the work_seconds array
        valid_last = ((eb >= 0) & (eb < num_bins))

        if np.any(valid_last):
            # get valid ending bins
            valid_eb = eb[valid_last]
            valid_e = e[valid_last]

            # if last time was 10:17, last bin start is 10:15
            last_bin_start = (timeline_start + valid_eb * bin_seconds)
            last_bin_seconds = (valid_e - last_bin_start)

            # add seconds between 10:15 and 10:17
            np.add.at(work_seconds,valid_eb, last_bin_seconds)

        # determines which requests have middle bins
        has_middle_bins = ((eb - sb) > 1)

        if np.any(has_middle_bins):

            middle_sb = sb[has_middle_bins]
            middle_eb = eb[has_middle_bins]

            # creates array of 0's
            difference = np.zeros(num_bins + 1, dtype=np.float64)

            # [0, 300, 0, ...]
            np.add.at(difference, middle_sb + 1,bin_seconds)
            #  [0, 300, 0, -300, 0 ]
            np.add.at(difference, middle_eb, -bin_seconds)

            # calculates cumulative sums
            work_seconds += np.cumsum(difference[:-1])

    # create final dataframe
    return pd.DataFrame({
        "bin_start": bin_starts,
        "bin_end": bin_ends,
        "work_seconds": work_seconds,
    })

def generate_workload_profile(dataset,selected_deployment, bin_size_minutes=5):
    """
    Convert raw LLM requests into a time-binned workload profile.

    This is the main workload-processing pipeline:

        Raw requests -> Calculate request durations -> Aggregate request activity into time bins -> Return workload profile
    """
    dataset = dataset.copy()

    # calculate start/end times for every request
    processed_df = calculate_duration(dataset, selected_deployment)

    # aggregate all request intervals into fixed time bins
    workload_profile = aggregate_workload(processed_df, bin_size_minutes=bin_size_minutes)

    return workload_profile

# ---------------------------------------------------------------------------
# SERVER POWER MODEL
# ---------------------------------------------------------------------------
def calculate_server_power(workload_profile, num_servers, selected_deployment, target_peak_util = 0.8):
    """
    Convert workload activity into server and cluster IT power.The workload profile is first normalized relative to its peak.
    That normalized activity is then scaled by the desired maximum utilization.

    Parameters
    ----------
    workload_profile : pandas.DataFrame, Binned workload containing "work_seconds".

    num_servers : int, Number of servers in the cluster.

    selected_deployment : str, Server configuration to use.

    target_peak_util : float
        Maximum utilization represented by the observed workload peak.
        For example, 0.8 means the trace peak corresponds to 80% utilization.

    Returns
    -------
    pandas.DataFrame
        Workload and calculated server/cluster power metrics.
    """

    spec = SERVER_CONFIGS[selected_deployment]
    df = workload_profile.copy()

    # find peak work seconds across the trace
    peak_trace_work = df["work_seconds"].max()

    # compute relative activity profile [0.0 to 1.0]
    df["relative_activity"] = df["work_seconds"] / peak_trace_work if peak_trace_work > 0 else 0.0

    # scale utilization according to the cluster's target design peak
    # target peak utilization defines the max expected workload capacity of a system under peak demand (80%)
    df["utilization"] = (df["relative_activity"] * target_peak_util).clip(0.0, 1.0)

    # Linear power model
    p_idle = spec["P_idle"]
    p_dynamic = spec["P_peak"] - spec["P_idle"]

    df["server_power_kw"] = (p_idle + p_dynamic * df["utilization"]) / 1000.0

    # Power of entire cluster
    df["it_power_kw"] = (num_servers * df["server_power_kw"])

    return df[
        [
            "bin_start",
            "bin_end",
            "work_seconds",
            "relative_activity",
            "utilization",
            "server_power_kw",
            "it_power_kw",
        ]
    ]

def calculate_power_profile(selected_deployment, num_servers, bin_size_minutes=5, target_peak_util=.8):

    # Load LLM workload
    day = random.randint(1, 7)

    filename = f"llm_inference_files/AzureLLMInferenceTrace_day{day}.csv"
    dataset = pd.read_csv(filename)

    # Convert requests into execution intervals
    workload_profile = generate_workload_profile(dataset, selected_deployment, bin_size_minutes=bin_size_minutes)

    power_profile = calculate_server_power(workload_profile, num_servers, selected_deployment, target_peak_util=target_peak_util)

    power_profile["timestamp"] = (power_profile["bin_start"])

    power_profile["hour"] = (
        (
            power_profile["timestamp"]
            - power_profile["timestamp"].min()
        ).dt.total_seconds()
        / 3600.0
    )

    return power_profile[
        [
            "timestamp",
            "hour",
            "work_seconds",
            "relative_activity",
            "utilization",
            "server_power_kw",
            "it_power_kw"
        ]
    ]


# def plot_24hr_power_profile(power_df, interval_num=5):
#     """Plots 24-hour electrical power load (kW) and  utilization U(t)
#     at 5-minute interval granularity.
#     """
#     # Ensure dataframe is reset to a clean integer index
#     df = power_df.reset_index(drop=True)

#     plt.figure(figsize=(12, 6))

#     # Plot the main Linear Power load curve
#     plt.plot(
#         df.index,
#         df['power_kw_linear'],
#         color='#1f77b4',
#         linewidth=2,
#         marker='o',
#         markersize=3,
#         label=f'{interval_num}-Min Linear Power Model',
#     )

#     # Format the X-Axis to display time ticks every 2 hours
#     total_bins = (24 * 60) // interval_num  # 288 bins for 5-min intervals
#     bins_per_hour = 60 // interval_num  # 12 bins per hour
#     tick_step = bins_per_hour * 2  # 24 bins per 2-hour step

#     tick_intervals = list(range(0, total_bins, tick_step))
#     tick_labels = [f'{(i * interval_num) // 60:02d}:00' for i in tick_intervals]

#     plt.xticks(ticks=tick_intervals, labels=tick_labels, rotation=0)
#     plt.xlim(0, total_bins - 1)

#     # Axis labels & Title
#     plt.title(
#         f'24-Hour Azure LLM Cluster Power Load Profile ({interval_num}-Minute'
#         ' Resolution)',
#         fontsize=14,
#         pad=15,
#     )
#     plt.xlabel('Time of Day (HH:MM)', fontsize=11, labelpad=10)
#     plt.ylabel('Cluster Electrical Load (kW)', fontsize=11, labelpad=10)

#     # Styling details
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend(loc='upper right')
#     plt.tight_layout()

#     # Display the plot
#     plt.show()


def main():
    """Main execution block where workflow functions are called."""
    # decrease_dataset()

    # Example execution: 10,000 servers in a Hyperscale Cloud DC
    # profile = calculate_power_profile(
    #     selected_deployment="Dense",
    #     num_servers=10000,
    #     bin_size_minutes=5,
    #     target_peak_util=0.85
    # )

    # print("\n--- Power Profile Sample (First 5 Bins) ---")
    # print(profile[["timestamp", "relative_activity", "utilization", "server_power_kw", "it_power_kw"]].head())

    # calculate_power_profile("Standard", num_servers=10, bin_size_minutes=5)

# --- RUN SCRIPT ---
if __name__ == "__main__":
    main()
