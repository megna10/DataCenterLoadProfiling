import pandas as pd
import numpy as np

int_active_vms = pd.read_csv('vmtable_filtered.csv')
# grabs every entry in the first column & converts IDs into a set
target_vm_ids = set(int_active_vms.iloc[:, 0])

# time in seconds modeling one day
START_TIME = 0
END_TIME = 86400

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

# rounds every decimal to nearest integer below it to pool cpu_util into hours
# adds a new column named hour into data frame
extracted_data["hour"] = np.floor(extracted_data['timestamp'] / 3600).astype(int)

# groups all rows in dataset into 24 distinct buckets
# calculate Hourly Average CPU Profile across all selected VMs
hourly_profile = extracted_data.groupby('hour')['avg_cpu'].mean().reset_index()

hourly_profile.to_csv('hourly_cpu_profiles_1_to_7.csv', index = False)
print(hourly_profile)
