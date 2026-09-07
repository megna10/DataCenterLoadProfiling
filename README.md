**Data Center Load Profiling**

A Python-based framework for generating 24-hour data center electrical load profiles from realistic workload traces. The project models how different data center workloads translate into server utilization, IT power consumption, and total facility power. It supports multiple data center environments, hardware deployment tiers, workload types, and Power Usage Effectiveness (PUE) values. 
The project also includes an interactive Streamlit-based data center power simulator for configuring and visualizing data center power demand.

**Overview**

Data center electricity demand is strongly influenced by the workload running on the underlying servers. Different workloads can produce very different temporal power profiles. This project converts workload traces into estimated electrical load profiles using the following pipeline:

Workload Trace -> Workload Activity -> Server Utilization -> Server IT Power -> Cluster IT Power -> PUE -> Total Facility Power

**Data Center Types**
The simulator supports several facility categories:
**Enterprise
Co-location
Hyperscale Cloud
Hyperscale AI
Edge**

Each facility type supports different combinations of workloads and deployment configurations.

**Supported Workloads**
**1) General Purpose Compute**
**2) Storage**
**3) Data Analytics / Batch Processing**
**4) AI Inference**
**5) AI Training**

**Deployment Tiers**

Different hardware deployment tiers are used to represent different server configurations. Depending on the workload, the available tiers include:
**Standard
Dense
Extreme**

Each deployment tier has its own hardware characteristics, including:
Idle power
Peak power
Number of GPUs or CPU cores
Workload-specific performance assumptions

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

𝑃
𝐼
𝑇
=
𝑁
×
𝑃
𝑠
𝑒
𝑟
𝑣
𝑒
𝑟

Total facility power is then calculated using Power Usage Effectiveness:

𝑃
𝑓
𝑎
𝑐
𝑖
𝑙
𝑖
𝑡
𝑦
=
𝑃
𝐼
𝑇
×
𝑃
𝑈
𝐸

where PUE represents additional facility overhead associated with cooling, power distribution, lighting, and other non-IT infrastructure.

Workload Normalization

Raw workload traces are normalized into a relative activity profile.

For a workload metric 
𝑊
(
𝑡
)
:

𝐴
(
𝑡
)
=
𝑊
(
𝑡
)
max
⁡
(
𝑊
)

The activity profile can then be scaled to a target peak utilization:

𝑈
(
𝑡
)
=
𝐴
(
𝑡
)
×
𝑈
𝑝
𝑒
𝑎
𝑘

This allows workload traces with different absolute magnitudes to be converted into comparable server utilization profiles.

Time Resolution

Most workload profiles are generated at 5-minute resolution.

A 24-hour profile therefore contains:

24
×
60
/
5
=
288

time intervals.

This provides enough temporal resolution to capture workload fluctuations while keeping the resulting datasets manageable.

Repository Structure
DataCenterLoadProfiling/
│
├── app.py
│
├── hourly_profile_general_compute.py
├── hourly_profile_storage.py
├── hourly_profile_data_analytics.py
├── hourly_profile_llm_inference.py
├── hourly_profile_llm_training.py
│
├── general_compute_files/
│   └── Workload data and intermediate files
│
├── storage_files/
│   └── Storage workload data
│
├── data_analytics_files/
│   └── Analytics workload data
│
├── llm_inference_files/
│   └── LLM inference traces
│
├── ai_training_files/
│   └── GPU training metrics and processed profiles
│
└── .gitignore

Main Components
app.py

The main interactive Streamlit application.

It provides controls for:

Data center type
Workload type
Deployment tier
Number of servers
PUE

The application calculates and displays:

Average facility power
Peak facility power
Minimum facility power
Peak time
Daily energy consumption
IT power
Facility power
Server utilization
24-hour power profile
hourly_profile_general_compute.py

Generates power profiles for conventional CPU-based compute workloads.

The model converts core-level workload activity into server utilization and then estimates server and cluster power.

hourly_profile_storage.py

Generates power profiles for storage-oriented workloads.

hourly_profile_data_analytics.py

Processes data analytics and batch-processing workloads and converts workload activity into an electrical load profile.

hourly_profile_llm_inference.py

Models LLM inference workloads.

Inference requests are converted into execution durations using estimated prefill and decode throughput:

𝑇
𝑟
𝑒
𝑞
𝑢
𝑒
𝑠
𝑡
=
𝑇
𝑜
𝑘
𝑒
𝑛
𝑠
𝑖
𝑛
𝑝
𝑢
𝑡
𝑅
𝑝
𝑟
𝑒
𝑓
𝑖
𝑙
𝑙
+
𝑇
𝑜
𝑘
𝑒
𝑛
𝑠
𝑜
𝑢
𝑡
𝑝
𝑢
𝑡
𝑅
𝑑
𝑒
𝑐
𝑜
𝑑
𝑒

The resulting request intervals are aggregated into time bins to produce a workload activity profile.

hourly_profile_llm_training.py

Processes GPU utilization traces from AI training infrastructure.

The model extracts GPU utilization from individual machines, creates 24-hour profiles, aggregates machines into a cluster-level profile, and converts GPU utilization into server power.

Interactive Simulator

The easiest way to explore the models is through the Streamlit application.

After installing the required Python packages, run:

streamlit run app.py


The application will open in a browser.

The user can configure:

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


The application then produces an interactive 24-hour power profile.

Installation

Clone the repository:

git clone https://github.com/megna10/DataCenterLoadProfiling.git
cd DataCenterLoadProfiling


Create a virtual environment.

Windows
python -m venv venv
venv\Scripts\activate

macOS / Linux
python3 -m venv venv
source venv/bin/activate


Install the required packages:

pip install numpy pandas scipy matplotlib plotly streamlit tqdm

Running the Application

Start the Streamlit application:

streamlit run app.py


Then open the local Streamlit URL displayed in the terminal.

Example Scenario

Suppose a user wants to model a Hyperscale AI facility running an AI Inference workload.

They can configure:

Data Center:
    Hyperscale AI

Workload:
    AI Inference

Deployment:
    Dense

Servers:
    10,000

PUE
