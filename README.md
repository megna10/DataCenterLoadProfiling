## Data Center Load Profiling

A Python-based framework for generating 24-hour data center electrical load profiles from realistic workload traces. The project models how different data center workloads translate into server utilization, IT power consumption, and total facility power. It supports multiple data center environments, hardware deployment tiers, workload types, and Power Usage Effectiveness (PUE) values. 
The project also includes an interactive Streamlit-based data center power simulator for configuring and visualizing data center power demand.

---


### Overview

Data center electricity demand is strongly influenced by the workload running on the underlying servers. Different workloads can produce very different temporal power profiles. This project converts workload traces into estimated electrical load profiles using the following pipeline:

Workload Trace -> Workload Activity -> Server Utilization -> Server IT Power -> Cluster IT Power -> PUE -> Total Facility Power

---


### Data Center Types
The simulator supports several facility categories:
**Enterprise
Co-location
Hyperscale Cloud
Hyperscale AI
Edge**

Each facility type supports different combinations of workloads and deployment configurations.

---


### Supported Workloads
**1) General Purpose Compute**
**2) Storage**
**3) Data Analytics / Batch Processing**
**4) AI Inference**
**5) AI Training**

---


### Deployment Tiers

Different hardware deployment tiers are used to represent different server configurations. Depending on the workload, the available tiers include:
**Standard
Dense
Extreme**

Each deployment tier has its own hardware characteristics, including:
Idle power
Peak power
Number of GPUs or CPU cores
Workload-specific performance assumptions

---


###  Power Modeling

The project utilizes a simplified linear server power model to calculate electrical draw.

#### 1. Individual Server Power
For a server with idle power $P_{\text{idle}}$, peak power $P_{\text{peak}}$, and utilization $U \in [0, 1]$:

$$P_{\text{server}} = P_{\text{idle}} + (P_{\text{peak}} - P_{\text{idle}}) \times U$$

Where:
* **$P_{\text{idle}}$:** Server power consumption at $0\%$ utilization ($W$)
* **$P_{\text{peak}}$:** Server power consumption at $100\%$ utilization ($W$)
* **$U$:** Normalized server compute utilization ($0.0 \text{ to } 1.0$)

---

#### 2. Cluster IT Power
For a cluster containing $N$ identical servers:

$$P_{\text{IT}} = N \times P_{\text{server}}$$

---

#### 3. Total Facility Power
Total data center facility power is calculated using **Power Usage Effectiveness (PUE)** to account for non-IT overhead:

$$P_{\text{facility}} = P_{\text{IT}} \times \text{PUE}$$

> **Note:** $\text{PUE}$ accounts for supporting infrastructure including cooling, power distribution losses, and lighting.

---


### Repository Structure
```text
DataCenterLoadProfiling/
├── app.py
├── hourly_profile_general_compute.py 
├── hourly_profile_storage.py
├── hourly_profile_data_analytics.py
├── hourly_profile_llm_inference.py
├── hourly_profile_llm_training.py
├── general_compute_files/
│   └── Workload data and intermediate files
├── storage_files/
│   └── Storage workload data
├── data_analytics_files/
│   └── Analytics workload data
├── llm_inference_files/
│   └── LLM inference traces
├── ai_training_files/
│   └── GPU training metrics and processed profiles
└── .gitignore
```
---

## Prerequisites & Installation

1) Clone the repository
   git clone https://github.com/megna10/DataCenterLoadProfiling.git
   cd DataCenterLoadProfiling
   
2) Install all necessary dependencies for core computations, data processing, statistical fitting, and the `app.py` dashboard using `pip`:
```bash
pip install numpy pandas scipy matplotlib streamlit
```

3) After installing the required Python packages, run in the terminal:

```bash
streamlit run app.py
```
The application will open in a browser. 
The user can configure:

```bash
Data Center Type
        ↓
Workload
        ↓
Deployment Tier
        ↓
Number of Servers
        ↓
PUE
        ↓
Calculate Power
```
The application then produces an interactive 24-hour power profile.

