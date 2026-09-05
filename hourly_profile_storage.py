import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Storage Server Configurations (Full System Specs)
SERVER_CONFIGS = {
    'Standard': {
        'max_iops': 16200000,  # Peak system IOPS rating
        'p_idle': 213,     # Total system Idle Watts
        'p_peak': 724,     # Total system Peak Watts
        'num_drives': 12,
    },
    'Dense': {
        'max_iops': 17232,
        'p_idle': 526,
        'p_peak': 1707.0,
        'num_drives': 24,
    },
    'Extremes': {
        'max_iops': 32400000,
        'p_idle': 560,
        'p_peak': 2279,
        'num_drives': 24,
    }
}

def read_data():
    """
    disk_attr = 0 or 1 → product/service-level category.
    disk_type = 0 or 1 → 0 = data disk, 1 = system disk.
    user_type = 0 or 1 → occasional vs. regular user.
    """
    columns = [
    "disk_uid",
    "disk_capacity",
    "disk_attr",
    "disk_type",
    "user_type",
    "vm_cpu",
    "vm_memory"
    ]

    df = pd.read_csv("disk_subscript_info", header=None, names=columns)

    df.to_csv("disk_subscription_info2.csv", index=False)

    print(df.head())

def get_only_data_disks(df):
    data_disks = df[df["disk_type"] == 0]
    data_disks.to_csv("data_disks.csv", index=False)

def get_only_attribute1(data_disks):
    data_disks_attr1 = data_disks[data_disks["disk_attr"] == 1]
    data_disks_attr1.to_csv("storage_files/data_disks_att1.csv", index=False)


def get_load_sample_data(data, sample_size=100, random_state=42):

    # randomly select 100 disks from the data set
    sample_disks = data.sample(n=sample_size, random_state=random_state)
    load_dir = "disk_load_data"

    # stores each disk's load data
    all_loads = []

    # Loop through the disk_uid of every randomly selected disk.
    for disk_uid in sample_disks["disk_uid"]:
        file_path = os.path.join(load_dir, disk_uid)

        load = pd.read_csv(file_path, header=None)

        load.columns = [
            "timestamp",
            "read_IOPS",
            "read_bandwidth",
            "write_IOPS",
            "write_bandwidth",
            "disk_usage"
        ]

        load["disk_uid"] = disk_uid

        all_loads.append(load)

    # Combine all 100 individual DataFrames into one large DataFrame.
    all_loads = pd.concat(all_loads, ignore_index=True)
    return all_loads

def get_24hr_profile(load):

    # converts unix timestamps into datetime values
    load["datetime"] = pd.to_datetime(load["timestamp"], unit="s")

    # Calculate total IOPS from read and write IOPS.
    load["total_IOPS"] = (load["read_IOPS"] + load["write_IOPS"])

    # store the 24 hour profiles for each disk
    disk_profiles = []

    # process each disk separately
    for disk_uid, disk_data in load.groupby("disk_uid"):

        # sort the disk's measurements chronologically
        disk_data = disk_data.sort_values("datetime").copy()

        # get first timestamp available for this disk
        start_time = disk_data["datetime"].min()

        end_time = start_time + pd.Timedelta(hours=24)

        # select only the requested 24 hour period
        day_data = disk_data[
            (disk_data["datetime"] >= start_time) &
            (disk_data["datetime"] < end_time)
        ].copy()

        # Only keep disks with a complete 24-hour period.
        if len(day_data) == 288:

            # Keep only the information we need.
            profile = day_data[
                ["datetime", "total_IOPS"]
            ].copy()

            # Create a 5-minute interval number from 0 to 287.
            profile["interval"] = range(288)

            # Store the disk ID.
            profile["disk_uid"] = disk_uid

            # Add this disk's profile to our list.
            disk_profiles.append(profile)

    # Combine all complete disk profiles. [date time, total iops, interval, disk id]
    profiles = pd.concat(
        disk_profiles,
        ignore_index=True
    )

    return profiles

def calculate_server_power(profile, selected_deployment, num_servers):
    """
    Converts storage workload into total storage server IT power.
    """
    spec = SERVER_CONFIGS[selected_deployment]

    df = profile.copy()

    # calcualte the workload of one server
    df["server_IOPS"] = (df["avg_disk_IOPS"] * spec["num_drives"])

    # convert iops into utilization.
    df["utilization"] = (df["server_IOPS"] / spec["max_iops"])

    # Prevent utilization from exceeding 100%.
    df["utilization"] = df["utilization"] = df["utilization"].clip(0.0, 1.0)

    # Linear server power model.
    p_idle = spec["p_idle"]
    p_dynamic = spec["p_peak"] - spec["p_idle"]

    df["server_power_kW"] = (
        p_idle +
        p_dynamic * df["utilization"]
    ) / 1000.0

    # Total power for all servers.
    df["it_power_kW"] = (
        df["server_power_kW"] * num_servers
    )

    return df[
        [
            "interval",
            "avg_disk_IOPS",
            "utilization",
            "server_power_kw",
            "power_kw"
        ]
    ]


def calculate_power_profile( selected_deployment, num_servers, sample_size=100, random_state=42):
    """
    Complete storage workload-to-power pipeline.
    """
     # Load selected storage disks.
    data_disks_attr1 = pd.read_csv(
        "storage_files/data_disks_att1.csv"
    )

    # Sample disks and load their IOPS traces.
    load = get_load_sample_data(
        data_disks_attr1,
        sample_size=sample_size,
        random_state=random_state
    )

    # Create 24-hour profile for each disk.
    profiles = get_24hr_profile(load)

    # Create representative workload.
    representative = create_representative_profile(profiles)

    # Convert workload into server power.
    power_profile = calculate_server_power(
        representative,
        selected_deployment,
        num_servers
    )

    result = power_profile.copy()

    result["hour"] = (
        result["interval"] * 5
    ) / 60.0

    # --------------------------------------------------------
    # Standardized output
    # --------------------------------------------------------

    return result[
        [
            "interval",
            "hour",
            "avg_disk_IOPS",
            "utilization",
            "server_power_kw",
            "it_power_kw",
        ]
    ]

def create_representative_profile(profiles):
    """calculate the avg IOPs across all sampled disks for 24hr profile for each 5 min interval"""

    representative = (
        profiles
        .groupby("interval")["total_IOPS"]
        .mean()
        .reset_index()
    )

    representative = representative.rename(columns={"total_IOPS": "avg_disk_IOPS"})
    return representative

def plot_profile(profile):
    plt.figure(figsize=(12, 5))

    plt.plot(
        profile["interval"],
        profile["server_power_kW"]
    )

    plt.xlabel("Time")
    plt.ylabel("Server Power (kW)")
    plt.title("Server Power Over 24 Hours")

    # One label per hour
    plt.xticks(
        range(0, 288, 12),
        [f"{i//12:02d}:00" for i in range(0, 288, 12)],
        rotation=45
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """Main execution block where workflow functions are called."""
    # df =  pd.read_csv('disk_subscription_info2.csv')

    # shows how much of each value appears
    # print(df["disk_attr"].value_counts())
    # print(df["disk_type"].value_counts())

    # capacity distribution for each disk_attr
    # print(df.groupby("disk_attr")["disk_capacity"].value_counts())

    # # show what disk types occur in disk attributes
    # print(df.groupby("disk_attr")["disk_type"].value_counts())

    # data_disks_attr1 = pd.read_csv("storage_files/data_disks_att1.csv")
    # load = get_load_sample_data(data_disks_attr1, 100, 42)

    # # create a 24 hour, 5 min profile for each disk
    # profiles = get_24hr_profile(load)

    # # Create the representative workload across the disks.
    # representative = create_representative_profile(profiles)
    # pro = calculate_power_profile(representative, 'Standard_Storage', 10)

    # plot_profile(pro)

    calculate_power_profile("Standard", 10, 100, 42 )


if __name__ == "__main__":
    main()
