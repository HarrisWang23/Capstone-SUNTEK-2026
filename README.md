# Design Patent Infringement Risk Assessment

**An LLM-powered system for evaluating potential design patent infringement risks through visual comparison.**

## Overview

This project was developed as part of the **2026 SUNTEK Capstone Project**.

The system uses a multimodal Large Language Model (LLM) to compare product images with design patent images and assess potential infringement risks. It also provides a Streamlit-based user interface for running evaluations and visualizing results.

## How It Works

The system follows four main steps:

1. **Data Input** — Load product images, patent images, and related product information from a JSON dataset.
2. **LLM Analysis** — Use LiteLLM to access a multimodal LLM that compares overall appearance, structural features, and visual similarities.
3. **Risk Assessment** — Generate a similarity score (0–100), a risk level (Low / Medium / High), and a brief explanation.
4. **Evaluation & Visualization** — Compare predictions with labeled data and display evaluation metrics and misclassified cases through a Streamlit dashboard.

## Getting Started

### Prerequisites

- Python 3.8+
- `litellm`
- `streamlit`
- `requests`

Install the required libraries:

```bash
pip install litellm streamlit requests
```

### Configuration

Configure the following environment variables before running the application:

| Variable | Description |
|---|---|
| `LITELLM_PROXY_API_KEY` | API key for accessing the LLM proxy |
| `LITELLM_PROXY_API_BASE` | Base URL of the LiteLLM proxy server |

The LLM model and proxy settings can be configured in `patent_agent.py`.

### Usage

After configuring the API and dataset paths, launch the user interface:

```bash
streamlit run ui_app.py
```

The dashboard allows users to select the number of samples and concurrent workers, run the assessment, and review evaluation results.

## Project Structure

```text
Capstone-SUNTEK-2026/
│
├── data/
│   ├── DPAS_RADAR_2.0.0-test.json
│   └── README.md
│
├── patent_agent.py    # LLM analysis and evaluation
├── ui_app.py          # Streamlit dashboard
├── client.py          # LiteLLM API example
└── README.md          # Project documentation
```

## Notes

- Due to company confidentiality restrictions, only limited sample data is provided for demonstration.
- The application requires valid access to the configured LLM proxy.
- The risk assessments are model-generated estimates and should not be considered formal legal conclusions.

---

*Developed as part of the SUNTEK Capstone Project, 2026.*
