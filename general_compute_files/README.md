General Compute Datasets:

Source: https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md

These large datasets are stored separately on Google Drive.

See the main project README for information about accessing the complete dataset.

    interactive_4cores.csv:
        filtered inventory table deining interactive VM instances equipped with 4 cores
        vm_id: matches active Azure virtual machines
        cores: num of CPU cores allocated per VM (4)
        ram: allocated system RAM (GB)

    target_vm_rows_1_7.csv:
        aggregated 24 hour trace extract pulled directly from azure data set. Dataset is filtered specifically to match target IDs in interactive_4cores.csv.
        timestamp: seconds(s)
        vm_id: vm idenitifier
        min_cpu
        max_cpu
        avg_cpu
