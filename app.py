import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

from hourly_profile_general_compute import calculate_power_profile as calculate_compute_power
from hourly_profile_storage import calculate_power_profile as calculate_storage_power
from hourly_profile_llm_inference import calculate_power_profile as calculate_llm_power
from hourly_profile_data_analytics import calculate_power_profile as calculate_analytics_power
POWER_MODELS = {
    'compute': calculate_compute_power,
    'storage': calculate_storage_power,
    'inference': calculate_llm_power,
    'analytics': calculate_analytics_power
}
# ==========================================
# 1. PAGE & STYLING CONFIGURATION
# ==========================================
st.set_page_config(
    page_title='Data Center Power Simulator', page_icon='⚡', layout='wide'
)

DATACENTER_TREE = {
    'Enterprise': {
        'General Purpose Compute': {
            'Standard': {'power_model': 'compute', 'default_pue': 1.40},
            'Dense': {'power_model': 'compute', 'default_pue': 1.35},
        },
        'Storage': {
            'Standard': {'power_model': 'storage', 'default_pue': 1.35},
            'Dense': {'power_model': 'storage', 'default_pue': 1.30},
        },
        'Data Analytics/Batch Processing': {
            'Standard': {'power_model': 'analytics', 'default_pue': 1.38},
            'Dense': {'power_model': 'analytics', 'default_pue': 1.38},
        },
        "AI Inference": {
            "Standard": {
                    "power_model": "inference",
                    "default_pue": 1.35},

},
    },
    'Co-location': {
        'General Purpose Compute': {
            'Standard': {'power_model': 'compute', 'default_pue': 1.25},
            'Dense': {'power_model': 'compute', 'default_pue': 1.22},
        },
        'Storage': {
            'Standard': {'power_model': 'storage', 'default_pue': 1.25},
            'Dense': {'power_model': 'storage', 'default_pue': 1.20},
        },
        'Data Analytics/Batch Processing': {
            'Standard': {'power_model': 'analytics', 'default_pue': 1.38},
            'Dense':{'power_model': 'analytics', 'default_pue': 1.38},
        },
        'AI Training': {
            'Standard': {'trace_type': 'synthetic_burst', 'default_pue': 1.28},
        },
        'AI Inference': {
            "Standard": {
                    "power_model": "inference",
                    "default_pue": 1.35}},
        'HPC': {
            'Standard': {'trace_type': 'borg_compute', 'default_pue': 1.25},
            'Dense': {'trace_type': 'borg_compute', 'default_pue': 1.22},
        },
    },
    'Hyperscale Cloud': {
        'General Purpose Compute': {
            'Standard': {'power_model': 'compute', 'default_pue': 1.15},
            'Dense': {'power_model': 'compute', 'default_pue': 1.12},
            'Extreme': {'power_model': 'compute', 'default_pue': 1.10},
        },
        'Storage': {
            'Standard': {'power_model': 'storage', 'default_pue': 1.18},
            'Dense': {'power_model': 'storage', 'default_pue': 1.15},
            'Extreme': {'power_model': 'storage', 'default_pue': 1.12},
        },
        'Data Analytics/Batch Processing': {
            'Standard': {'power_model': 'analytics', 'default_pue': 1.38},
            'Dense': {'power_model': 'analytics', 'default_pue': 1.38},
            'Extreme': {'power_model': 'analytics', 'default_pue': 1.38},
        },
        'AI Training': {
            'Standard': {'trace_type': 'synthetic_burst',
            'default_pue': 1.20},
            'Dense': {'trace_type': 'synthetic_burst', 'default_pue': 1.18},
            'Extreme': {'trace_type': 'synthetic_burst', 'default_pue': 1.15},
        },
        'AI Inference': {
            "Standard": {
                    "power_model": "inference",
                    "default_pue": 1.35},
            'Dense': {
                    "power_model": "inference",
                    "default_pue": 1.35},
            'Extreme': {
                    "power_model": "inference",
                    "default_pue": 1.35},
        },
        'HPC': {
            'Standard': {'trace_type': 'borg_compute', 'default_pue': 1.18},
            'Dense': {'trace_type': 'borg_compute', 'default_pue': 1.15},
            'Extreme': {'trace_type': 'borg_compute', 'default_pue': 1.12},
        },
    },
    'Hyperscale AI': {
        'AI Training': {
            'Standard': {'trace_type': 'synthetic_burst', 'default_pue': 1.18},
            'Dense': {'trace_type': 'synthetic_burst', 'default_pue': 1.15},
            'Extreme': {'trace_type': 'synthetic_burst', 'default_pue': 1.12},
        },
        'AI Inference': {
            'Dense': {
                    "power_model": "inference",
                    "default_pue": 1.35},
            'Extreme': {
                    "power_model": "inference",
                    "default_pue": 1.35},
        },
    },
    'Edge': {
        'General Purpose Compute': {
            'Standard': {'power_model': 'compute', 'default_pue': 1.50},
            'Dense': {'power_model': 'compute', 'default_pue': 1.45},
        },
        'Storage': {
            'Standard': {'power_model': 'storage', 'default_pue': 1.45},
            'Dense': {'power_model': 'storage', 'default_pue': 1.40},
        },
        'AI Inference': {
            "Standard": {
                    "power_model": "inference",
                    "default_pue": 1.35}
        },
    },
}


def calculate_selected_power(metadata, selected_deployment, num_servers):
    power_model = metadata["power_model"]

    if power_model not in POWER_MODELS:
        raise ValueError(f"Unknown power model: {power_model}")

    calculate_power = POWER_MODELS[power_model]

    return calculate_power(
        selected_deployment=selected_deployment,
        num_servers=num_servers,
    )

st.sidebar.header('Facility Setup')

# -------------------------------------------------------------
# STEP 1: Select Data Center Type
# -------------------------------------------------------------
dc_type = st.sidebar.selectbox(
    'Step 1: Data Center Type', options=list(DATACENTER_TREE.keys()), index=2
)

# -------------------------------------------------------------
# STEP 2: Select Workload (Filtered by DC Type)
# -------------------------------------------------------------
workload_options = list(DATACENTER_TREE[dc_type].keys())
selected_workload = st.sidebar.selectbox(
    'Step 2: Workload Option', options=workload_options
)

# -------------------------------------------------------------
# STEP 3: Select Deployment Variant (Standard / Dense / Extreme)
# -------------------------------------------------------------
deployment_options = list(
    DATACENTER_TREE[dc_type][selected_workload].keys()
)
selected_deployment = st.sidebar.selectbox(
    'Step 3: Deployment Type', options=deployment_options
)

# Extract scenario metadata & hardware tier mapping
metadata = DATACENTER_TREE[dc_type][selected_workload][selected_deployment]

# -------------------------------------------------------------
# STEP 4: User Inputs Number of Servers
# -------------------------------------------------------------
st.sidebar.markdown('---')

num_servers = st.sidebar.number_input(
    'Step 4: Number of Servers (N)',
    min_value=1,
    max_value=1000000,
    value=100,
    step=10,
)

# -------------------------------------------------------------
# STEP 5: Facility Overhead & Utility Rates
# -------------------------------------------------------------
st.sidebar.markdown('---')
st.sidebar.subheader('Facility Parameters')

# PUE defaults automatically based on DC type + Deployment, but stays editable
pue = st.sidebar.slider(
    'Power Usage Effectiveness (PUE)',
    min_value=1.0,
    max_value=2.0,
    value=metadata['default_pue'],
    step=0.01,
    help='Auto-filled baseline for this facility tier.',
)

# ============================================================
# CALCULATE POWER
# ============================================================


if st.button("Calculate Power"):

    power_profile = calculate_selected_power(
        metadata=metadata,
        selected_deployment=selected_deployment,
        num_servers=num_servers,
    )

    st.session_state["power_profile"] = power_profile

if "power_profile" not in st.session_state:

    st.info("Configure the data center and click Calculate Power.")

    st.stop()

power_df = st.session_state["power_profile"].copy()
# ============================================================
# FACILITY POWER
# ============================================================

power_df = calculate_selected_power(metadata=metadata, selected_deployment=selected_deployment, num_servers=num_servers).copy()

power_df["facility_power_kw"] = (
    power_df["it_power_kw"] * pue
)


# ============================================================
# BASIC TIME INFORMATION
# ============================================================

if "hour" not in power_df.columns:

    if "timestamp" in power_df.columns:

        timestamp = pd.to_datetime(
            power_df["timestamp"]
        )

        power_df["hour"] = (
            timestamp - timestamp.min()
        ).dt.total_seconds() / 3600

    else:

        power_df["hour"] = (
            range(len(power_df))
        )


# ============================================================
# POWER STATISTICS
# ============================================================

it_power = power_df["it_power_kw"]

facility_power = power_df[
    "facility_power_kw"
]

average_power = facility_power.mean()

peak_power = facility_power.max()

minimum_power = facility_power.min()

peak_index = facility_power.idxmax()

peak_time = power_df.loc[
    peak_index,
    "hour"
]

# ============================================================
# ENERGY
# ============================================================

if len(power_df) > 1:

    interval_hours = (
        power_df["hour"].iloc[1]
        - power_df["hour"].iloc[0]
    )

else:

    interval_hours = 5 / 60


energy_kwh = (
    facility_power.sum()
    * interval_hours
)




st.header("Data Center Power Simulation")

st.caption(
    f"{dc_type} • {selected_workload} • "
    f"{selected_deployment} • {num_servers:,} servers"
)


# ============================================================
# KPI ROW
# ============================================================
st.subheader("Power Summary")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Average Power",
        f"{average_power:,.1f} kW"
    )

with col2:

    st.metric(
        "Peak Power",
        f"{peak_power:,.1f} kW"
    )

with col3:

    st.metric(
        "Daily Energy",
        f"{energy_kwh:,.2f} kWh"
    )


# ============================================================
# SECOND KPI ROW
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Minimum Power",
        f"{minimum_power:,.1f} kW"
    )

with col2:

    st.metric(
        "Peak Time",
        f"{peak_time:.2f} hr"
    )

with col3:

    st.metric(
        "PUE",
        f"{pue:.2f}"
    )

with col4:

    st.metric(
        "Servers",
        f"{num_servers:,}"
    )

# ============================================================
# POWER PROFILE
# ============================================================

st.subheader(
    "24-Hour Power Profile"
)

fig = go.Figure()


# IT POWER

fig.add_trace(
    go.Scatter(
        x=power_df["hour"],
        y=it_power,
        mode="lines",
        name="IT Power",
        line=dict(
            color="#2563EB",
            width=3,
        ),
        hovertemplate=(
            "<b>Time:</b> %{x:.2f} hr<br>"
            "<b>IT Power:</b> %{y:,.2f} kW"
            "<extra></extra>"
        ),
    )
)


# FACILITY POWER

fig.add_trace(
    go.Scatter(
        x=power_df["hour"],
        y=facility_power,
        mode="lines",
        name="Facility Power",
        line=dict(
            color="#F97316",
            width=3,
        ),
        hovertemplate=(
            "<b>Time:</b> %{x:.2f} hr<br>"
            "<b>Facility Power:</b> %{y:,.2f} kW"
            "<extra></extra>"
        ),
    )
)


fig.update_layout(
    height=550,
    template="plotly_white",
    hovermode="x unified",

    xaxis_title="Time of Day",
    yaxis_title="Power (kW)",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),

    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20,
    ),
)


fig.update_xaxes(
    tickmode="array",
    tickvals=list(range(0, 25, 2)),
    ticktext=[
        f"{hour:02d}:00"
        for hour in range(0, 25, 2)
    ],
)


st.plotly_chart(
    fig,
    use_container_width=True,
)

# ============================================================
# UTILIZATION
# ============================================================

if "utilization" in power_df.columns:

    st.subheader(
        "Server Utilization"
    )

    utilization_fig = go.Figure()

    utilization_fig.add_trace(
        go.Scatter(
            x=power_df["hour"],
            y=power_df["utilization"] * 100,
            mode="lines",
            name="Utilization",
            line=dict(
                color="#16A34A",
                width=2,
            ),
            fill="tozeroy",
            hovertemplate=(
                "<b>Time:</b> %{x:.2f} hr<br>"
                "<b>Utilization:</b> %{y:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    utilization_fig.update_layout(
        height=350,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Time of Day",
        yaxis_title="Utilization (%)",
        yaxis=dict(
            range=[0, 100]
        ),
    )

    st.plotly_chart(
        utilization_fig,
        use_container_width=True,
    )

# ============================================================
# CONFIGURATION
# ============================================================

st.subheader(
    "Configuration"
)

config_col1, config_col2 = st.columns(2)

with config_col1:

    st.write(
        {
            "Data Center": dc_type,
            "Workload": selected_workload,
            "Deployment": selected_deployment,
            "Servers": num_servers,
        }
    )

with config_col2:

    st.write(
        {
            "Power Model": metadata["power_model"],
            "PUE": pue,
            "Average Facility Power":
                f"{average_power:,.2f} kW",
            "Peak Facility Power":
                f"{peak_power:,.2f} kW",
        }
    )
