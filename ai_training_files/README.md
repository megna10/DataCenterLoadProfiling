AI Training Data Files:

This folder contains data files used by the AI training analysis

    8_gpu_machine_ids.csv:
        Contains the machine IDs corresponding to machines equipped with 8 GPUs since the server configurations all contain 8 GPUs. The machine IDs in this file are matched against the machine_id field in the machine metric datasets. This allows the analysis to restrict GPU utilization and power calculations to machines with the desired hardware configuration.


    cluster_gpu_profile.csv:
        This is a processed GPU utilization profile generated from the machine-level metric data. The data represents GPU utilization across a 24-hour period using 5-minute time intervals. This file contains approx. 288 time intervals where for each interval, GPU utilization measurements from the selected 8-GPU machine are aggregated to produce a cluster level util. profile.

Large Data Files:

\*pai files from https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020 , this includes more info about parameters

These include:

    pai_machine_metric.csv:
        original machine-level metric data.

    pai_machine_spec.csv:
        contains machine hardware configurations that allows us to identify machine that contain a specific number of GPUs

    metric_clean.csv:
        machine-level metric data without empty slots that contains measurements associated with individual machines over time.

    metric_8gpu.csv:
        filtered version of metric_clean.csv that contains only machine measurments with 8 GPUs.

The large datasets are stored separately on Google Drive.

See the main project README for information about accessing the complete dataset.

Data Processing

The general processing workflow is:

Original PAI metric data
↓
Identify 8-GPU machines
↓
Filter machine metrics
↓
Calculate 5-minute GPU utilization
↓
Aggregate across machines
↓
cluster_gpu_profile.csv (preprocessed file in order to spped up outputs)
↓
Server power calculation
