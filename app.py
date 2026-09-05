import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================
# 1. PAGE & STYLING CONFIGURATION
# ==========================================
st.set_page_config(
    page_title='Data Center Power Configurator', page_icon='⚡', layout='wide'
)

DATACENTER_TREE = {
    'Enterprise': {
        'General Purpose Compute': {
            'Standard': {'workload': 'borg_compute', 'default_pue': 1.40},
            'Dense': {'trace_type': 'borg_compute', 'default_pue': 1.35},
        },
        'Storage': {
            'Standard': {'trace_type': 'tencent_storage', 'default_pue': 1.35},
            'Dense': {'trace_type': 'tencent_storage', 'default_pue': 1.30},
        },
        'Data Analytics/Batch Processing': {
            'Standard': {'trace_type': 'borg_compute', 'default_pue': 1.38},
            'Dense': {'trace_type': 'borg_compute', 'default_pue': 1.32},
        },
        "AI Inference": {
    "Standard": {
        "model": "llm_inference",
        "trace": "azure_llm_inference",
        "gpu_options": ["L40S", "H100", "H200"],
        "default_gpu": "H100",
        "default_pue": 1.35,
    },

},
    },
    'Co-location': {
        'General Purpose Compute': {
            'Standard': {'trace_type': 'borg_compute', 'default_pue': 1.25},
            'Dense': {'trace_type': 'borg_compute', 'default_pue': 1.22},
        },
        'Storage': {
            'Standard': {'trace_type': 'tencent_storage', 'default_pue': 1.25},
            'Dense': {'trace_type': 'tencent_storage', 'default_pue': 1.20},
        },
        'Data Analytics/Batch Processing': {
            'Standard': {'trace_type': 'borg_compute', 'default_pue': 1.24},
            'Dense': {'trace_type': 'borg_compute', 'default_pue': 1.20},
        },
        'AI Training': {
            'Standard': {'trace_type': 'synthetic_burst', 'default_pue': 1.28},
        },
        'AI Inference': {
            "Standard": {
                    "model": "llm_inference",
                    "trace": "azure_llm_inference",
                    "gpu_options": ["L40S", "H100", "H200"],
                    "default_gpu": "H100",
                    "default_pue": 1.35,
        }},
        'HPC': {
            'Standard': {'trace_type': 'borg_compute', 'default_pue': 1.25},
            'Dense': {'trace_type': 'borg_compute', 'default_pue': 1.22},
        },
    },
    'Hyperscale Cloud': {
        'General Purpose Compute': {
            'Standard': {'trace_type': 'borg_compute', 'default_pue': 1.15},
            'Dense': {'trace_type': 'borg_compute', 'default_pue': 1.12},
            'Extreme': {'trace_type': 'borg_compute', 'default_pue': 1.10},
        },
        'Storage': {
            'Standard': {'trace_type': 'tencent_storage', 'default_pue': 1.18},
            'Dense': {'trace_type': 'tencent_storage', 'default_pue': 1.15},
            'Extreme': {'trace_type': 'tencent_storage', 'default_pue': 1.12},
        },
        'Data Analytics/Batch Processing': {
            'Standard': {'trace_type': 'borg_compute', 'default_pue': 1.15},
            'Dense': {'trace_type': 'borg_compute', 'default_pue': 1.12},
            'Extreme': {'trace_type': 'borg_compute', 'default_pue': 1.10},
        },
        'AI Training': {
            'Standard': {'trace_type': 'synthetic_burst', 'default_pue': 1.20},
            'Dense': {'trace_type': 'synthetic_burst', 'default_pue': 1.18},
            'Extreme': {'trace_type': 'synthetic_burst', 'default_pue': 1.15},
        },
        'AI Inference': {
            "Standard": {
                    "model": "llm_inference",
                    "trace": "azure_llm_inference",
                    "gpu_options": ["L40S", "H100", "H200"],
                    "default_gpu": "H100",
                    "default_pue": 1.35},
            'Dense': {'trace_type': 'synthetic_burst', 'default_pue': 1.12},
            'Extreme': {'trace_type': 'synthetic_burst', 'default_pue': 1.10},
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
            'Dense': {'trace_type': 'synthetic_burst', 'default_pue': 1.12},
            'Extreme': {'trace_type': 'synthetic_burst', 'default_pue': 1.08},
        },
    },
    'Edge': {
        'General Purpose Compute': {
            'Standard': {'trace_type': 'borg_compute', 'default_pue': 1.50},
            'Dense': {'trace_type': 'borg_compute', 'default_pue': 1.45},
        },
        'Storage': {
            'Standard': {'trace_type': 'tencent_storage', 'default_pue': 1.45},
            'Dense': {'trace_type': 'tencent_storage', 'default_pue': 1.40},
        },
        'AI Inference': {
            "Standard": {
                    "model": "llm_inference",
                    "trace": "azure_llm_inference",
                    "gpu_options": ["L40S", "H100", "H200"],
                    "default_gpu": "H100",
                    "default_pue": 1.35}
        },
    },
}

st.sidebar.header('Facility Setup Wizard')

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
    help=(
        'Specify the total node count deployed in this server cluster'
        ' configuration.'
    ),
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

electricity_rate = st.sidebar.number_input(
    'Electricity Rate ($ / kWh)',
    min_value=0.01,
    max_value=1.00,
    value=0.12,
    step=0.01,
)
