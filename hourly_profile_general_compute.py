import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# time in seconds modeling one day
START_TIME = 0
END_TIME = 86400

def get_target_vmid_table():
    int_active_vms = pd.read_csv('interactive_4cores.csv')
    # grabs every entry in the first column & converts IDs into a set
    target_vm_ids = set(int_active_vms.iloc[:, 0])

    # fetch the azure dataset directly from github
    links_url = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzurePublicDatasetLinksV2.txt"

    # streams the sites directly without downloading the links to the desktop
    links_df = pd.read_csv(links_url, header=None, names=['url'])

    # slices lines 12 through 18 by row index
    target_files_df = links_df.iloc[11:18]

    # extract the urls into a python list of strings
    target_files = [url.strip() for url in target_files_df['url']]

    filtered_chunks = []
    for url in target_files:
        # read directly from cloud URL
        # trace format: [timestamp, vm id, min_cpu, max_cpu, avg_cpu]
        df = pd.read_csv(url,
                        header=None,
                        names=['timestamp', 'vm_id', 'min_cpu', 'max_cpu', 'avg_cpu'],
                        compression='infer')

        # keep only target vm_ids (data frame only keeps true values)
        df_filtered = df[df['vm_id'].isin(target_vm_ids)].copy()
        filtered_chunks.append(df_filtered)

    # combines all extracted data into one table
    extracted_data = pd.concat(filtered_chunks, ignore_index=True)
    extracted_data.to_csv('target_vm_rows_1_7.csv', index=False)

    return extracted_data



def get_min_profiles(df_data, interval_min):

    # filters for day 1
    first_day_data = df_data[df_data['timestamp'] < END_TIME].copy()

    # rounds every decimal to nearest integer below it to pool cpu_util into hours
    # adds a new column named hour into data frame
    first_day_data[f"interval_{interval_min}m"] = (first_day_data['timestamp'] // (interval_min* 60)).astype(int)

    # groups all rows in dataset into 24 distinct buckets
    # calculate Hourly Average CPU Profile across all selected VMs
    profile = first_day_data.groupby(f'interval_{interval_min}m')['avg_cpu'].mean().reset_index()

    profile.to_csv(f'hourly_cpu_profiles_{interval_min}.csv', index = False)
    return profile


def get_vm_sections(max_core_count, num_groups):
    df = pd.read_csv("interactive_4cores.csv")

    df.columns = ["vm_id", "id2", "id3", "start_time", "stop_time", "max_cpu", "min_cpu", "avg_cpu","workload", "cores", "ram"]

    sections = []
    current_vm_ids = []
    current_cores = 0
    current_ram = 0
    total_group_count = 0

    for _, row in df.iterrows():
        cores = row["cores"]
        ram = row["ram"]

        # start a new section if adding this VM would exceed the limit
        if current_cores + cores > max_core_count:
            # each section is a dict inside a list
            sections.append({
                "vm_ids": current_vm_ids,
                "total_cores": current_cores,
                "total_ram": current_ram,
                })

            total_group_count += 1

            if total_group_count > num_groups:
                break

            # reset values
            current_ram = 0
            current_cores = 0
            current_vm_ids = []


        # update values
        current_vm_ids.append(row["vm_id"])
        current_cores += cores
        current_ram += ram

    # Add the final section / maybe it doesnt matter for now
    # if current_vm_ids:
    #     sections.append({
    #         "vm_ids": current_vm_ids,
    #         "total_cores": current_cores,
    #         "total_ram": current_ram
    #     })

    return sections


def get_avg_cpu_utilizations(sections, start_time, end_time):
    """
    sections: list of dictionaries containing groups of vm's

    """
    # Read the CPU utilization data
    df = pd.read_csv("target_vm_rows_1_7.csv")

    # Only use timestamps within the requested range
    df = df[
        (df["timestamp"] >= start_time) &
        (df["timestamp"] <= end_time)
    ]

    # will contain dicts of avg cpus of each section at different timestamps
    results = []

    # Go through each unique timestamp in order
    for timestamp in sorted(df["timestamp"].unique()):

        # Get all VM data for this timestamp
        timestamp_data = df[df["timestamp"] == timestamp]

        # print(f"\nTimestamp: {timestamp}")

        # creates a dict for this timestamp
        timestamp_result = {
            "timestamp": timestamp
        }

        # Go through each section
        for section_num, section in enumerate(sections, start=1):

            vm_ids = section["vm_ids"]

            # Find the rows for the VMs in this section
            section_data = timestamp_data[
                timestamp_data["vm_id"].isin(vm_ids)
            ]

            # Calculate the average CPU utilization
            if not section_data.empty:
                section_avg_cpu = section_data["avg_cpu"].mean()
            else:
                section_avg_cpu = None

            # Store the result
            timestamp_result[f"section_{section_num}_avg_cpu"] = section_avg_cpu

            # # Print result
            # if section_avg_cpu is not None:
            #     print(
            #          f"Timestamp={timestamp}, "
            #         f"  Section {section_num}: "
            #         f"Avg CPU = {section_avg_cpu:.2f}, "
            #         f"Total RAM = {section['total_ram']}"
            #     )
            # else:
            #     print(
            #         f"  Section {section_num}: "
            #         f"No CPU data, "
            #         f"Total RAM = {section['total_ram']}"
            #     )

        results.append(timestamp_result)

    return pd.DataFrame(results)

def get_overall_avg_cpu(section_results):
    """
    Calculate the average CPU utilization across all sections
    for each timestamp.

    section_results: DataFrame returned by get_avg_cpu_utilizations()

    """
    # Find all columns containing section CPU averages
    section_columns = [
        column for column in section_results.columns if column.startswith("section_") and column.endswith("_avg_cpu")
    ]

    # Calculate the average across sections for each timestamp
    section_results["overall_avg_cpu"] = (
        section_results[section_columns].mean(axis=1)
    )

    return section_results[["timestamp", "overall_avg_cpu"]]

# ------------- Plotting --------------------------

def plot_profiles(profile_df, interval_num):
    """ Plot the main CPU load curve"""

    plt.plot(
        profile_df[f'interval_{interval_num}m'],
        profile_df['avg_cpu'],
        color='#1f77b4',
        linewidth=2,
        marker='o',
        markersize=3,
        label=f'{interval_num}-Min Avg CPU Utilization'
    )

    # Format the X-Axis to display time ticks every 2 hours (every 8 15-min intervals)
    total_bins = (24 * 60) // interval_num
    bins_per_hour = 60 // interval_num          # e.g., 12 for 5-min, 4 for 15-min
    tick_step = bins_per_hour * 2

    tick_intervals = range(0, total_bins + 1, tick_step)
    tick_labels = [f"{(i * interval_num) // 60:02d}:00" for i in tick_intervals]

    plt.xticks(ticks=tick_intervals, labels=tick_labels, rotation=0)
    plt.xlim(0, total_bins - 1)

    # Axis labels & Title
    plt.title(f"24-Hour Azure VM CPU Load Profile ({interval_num}-Minute Resolution)", fontsize=14, pad=15)
    plt.xlabel("Time of Day (HH:MM)", fontsize=11, labelpad=10)
    plt.ylabel("Average CPU Utilization (%)", fontsize=11, labelpad=10)

    # Styling details
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Display the plot
    plt.show()

def plot_overall_cpu_util_sections(section_results):
    plt.figure(figsize=(12, 6))

    plt.plot(
        section_results["timestamp"],
        section_results["overall_avg_cpu"],
        linewidth=2,
        marker="o",
        markersize=3
    )


    plt.xlabel("Time (seconds)")
    plt.ylabel("Average CPU Utilization (%)")
    plt.title("Overall CPU Utilization Over Time")

    plt.grid(True)
    plt.show()


def main():
    """Main execution block where workflow functions are called."""
    # vm_id_table = get_target_vmid_table()
    # time_interval = 10
    # df_input = pd.read_csv('target_vm_rows_1_7.csv')
    # profile_df = get_min_profiles(df_input, time_interval)
    # plot_profiles(profile_df, time_interval)

    sects_of_ids = get_vm_sections(64, 10)
    df = get_avg_cpu_utilizations(sects_of_ids, 0, 86400 )
    plot_overall_cpu_util_sections(get_overall_avg_cpu(df))

# --- RUN SCRIPT ---
if __name__ == "__main__":
    main()
