Storage Datasets:

    disk_subscript_info file:
        unfiltered subscription metadata for all disk volume instances in the cluster
        disk_uid -> maps directly to filenames in disk_load_data
        disk_capacity -> storage volume capacity (GB)
        disk_attr -> 0 or 1 show product or service level storage
        disk_type -> 0 = data disk, 1 = system disk
        user_type ->  0 or 1 → occasional vs. regular user.
        vm_cpu -> core count of vm attached to this disk
        vm_memory -> memory allocation (GB) of VM attached to this disk

    disk_subscription_info2.csv:
        reads disk_subscript info and attaches explicit column headers

    data_disks.csv:
        filtered dataset that only contains only active volume instances dedicated to adata disks (disk_type == 0)

    data_disks_att1.csv:
        filtered dataset used as the base sampling population. gets only disk_attr = 1 since that has more values

These large files are stored separately on Google Drive.

See the main project README for information about accessing the complete dataset.

    disk_load_data/:
        folder that contains files of individual disks. each file contains timestamped performance traces recorded at 5 min intervals
        timestamp
        read_IOPS: num of read I/O operations per seconds
        read_bandwidth
        write_IOPS: num of write I/O operations per sec
        write_bandwidth
        disk_usage: total allocated disk capacity
