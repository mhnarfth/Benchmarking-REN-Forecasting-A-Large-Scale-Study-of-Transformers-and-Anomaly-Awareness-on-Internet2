# Benchmarking REN Traffic Forecasting: A Large-Scale Study on Internet2

**Author:** Mohammad Arafath Uddin Shariff; email: mshariff2@unl.edu

This repository contains the code and Jupyter Notebooks for a project focused on **scalable and anomaly-aware forecasting of Research and Education Network (REN) traffic**. It provides an end-to-end pipeline to process large Netflow datasets, integrate anomaly detection, and benchmark various time series forecasting models.

## Project Idea

The project aims to develop a robust framework for predicting network traffic in REN environments, which are characterized by unique "elephant flows" and burstiness. A key aspect is investigating how network anomalies (like sudden traffic surges) can be detected and integrated into forecasting models to improve prediction accuracy and operational resilience. The pipeline is designed to be scalable across multiple routers and computationally efficient.

## Dataset

The study uses Internet2 Netflow data from **October 8th to December 3rd, 2021**. This dataset comprises individual flow records (over 300 GB raw) from **10 backbone routers**. The initial raw data format (e.g., compressed files) is first converted into individual Parquet files for each router, which then serve as the input for the main ETL pipeline.

## Repository Structure

The core workflow is divided into three sequential Jupyter Notebooks, supported by a prerequisite data ingestion notebook, utility notebooks, and an `outputs/` directory for all generated artifacts.

```
.
├── NB1-ETL-pipeline.ipynb                  # Main Execution Notebook 1: Raw Data ETL & Anomaly Flagging
├── NB2-CPU-Models-without-patchTST.ipynb   # Main Execution Notebook 2: CPU-Friendly Models Training & Evaluation
├── NB3-GPU-models-patchTST.ipynb           # Main Execution Notebook 3: GPU/Advanced Models Training & Final Visualizations
│
├── Raw Data Ingestion, Preprocessing.ipynb # PREREQUISITE: Converts initial raw data (e.g., .dzip) into individual .parquet files
├── architecture_builder.ipynb              # Supplemental: Generates model architecture diagrams
├── EDA_visualization.ipynb                 # Supplemental: For additional, in-depth EDA
├── requirements.txt                        # Python dependencies for the project
├── Result-visualization.ipynb              # Supplemental: For custom post-analysis and visualization
│
└── outputs/                                # ALL generated output files are stored here
    ├── architectures/                      # Model architecture diagrams
    ├── ch4/                                # Main output folder for Chapter 4 results
    │   ├── figs/                           # Generated plots (PNG and PDF)
    │   ├── predictions/                    # Saved test set predictions (y_true, y_pred)
    │   ├── hourly_<router>_processed_with_anomalies.parquet # 10 files: Processed hourly data per router
    │   ├── all_model_results.csv           # Consolidated performance metrics for all runs
    │   └── ...
    └── ch5/                                # Placeholder for Chapter 5 outputs
        └── ...
```

## Setup & Dependencies

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/mhnarfth/Benchmarking-REN-Forecasting-A-Large-Scale-Study-of-Transformers-and-Anomaly-Awareness-on-Internet2.git
    cd Benchmarking-REN-Forecasting-A-Large-Scale-Study-of-Transformers-and-Anomaly-Awareness-on-Internet2
    ```

2.  **Create a Python Virtual Environment (Highly Recommended):**
    ```bash
    python -m venv my_network_env
    source my_network_env/bin/activate  # On Windows: .\my_network_env\Scripts\activate
    ```

3.  **Install Dependencies:**
    All required packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *   **PyTorch Note:** Notebooks are CPU-configured. For GPU, after installing from `requirements.txt`, you might need to install `torch` separately with CUDA support (check [PyTorch official instructions](https://pytorch.org/get-started/locally/) for your specific CUDA version) if the `requirements.txt` lists the CPU version by default.

4.  **Data Location:**
    *   **Initial Raw Data:** Identify the directory where your *original raw data files* (e.g., `.dzip` files mentioned in earlier context) are located. This will be the input for `Raw Data Ingestion, Preprocessing.ipynb`.
    *   **Processed Raw Parquet Files:** `Raw Data Ingestion, Preprocessing.ipynb` is expected to output individual router Parquet files (e.g., `atlanta.parquet`) into the directory specified by `INPUT_ROUTERS_DIR` in `NB1-ETL-pipeline.ipynb` (e.g., `/mnt/nrdstor/ramamurthy/mhnarfth/internet2/parquet`).

## How to Run the Pipeline (End-to-End)

The full Chapter 4 pipeline executes sequentially. **It is critical to run the notebooks in the specified order: Prerequisite, then NB1, then NB2, then NB3.**

**Important Considerations:**
*   **Validation Run:** `EPOCHS` for neural network models are set to `2` for a quick validation run. This is for pipeline verification, not optimal performance.
*   **Resource Management:** The pipeline is optimized for 60 GB RAM / 16 CPU cores. Memory-intensive raw data processing is managed by Dask and sequential per-router processing.
*   **Intermediate Outputs:** Each notebook saves processed data and results in `outputs/ch4/`, which are consumed by subsequent notebooks.

---

### **Prerequisite Notebook: `Raw Data Ingestion, Preprocessing.ipynb`**
*(Converts raw data to individual Parquet files)*

**Purpose:** This notebook handles the very first stage of data preparation. It takes your original raw data files (e.g., `.dzip` or custom JSON) and converts them into the individual `*.parquet` files (e.g., `atlanta.parquet`, `dallas.parquet`) that `NB1-ETL-pipeline.ipynb` expects as input.

1.  Open `Raw Data Ingestion, Preprocessing.ipynb`.
2.  **Review its `Configuration`** to ensure input paths (for original raw data) and output paths (for individual `*.parquet` files) are correctly set.
3.  Run all cells sequentially.

**Expected Output:** Individual `*.parquet` files (one per router) created in the specified `INPUT_ROUTERS_DIR`.

---

### **Step 1: Execute `NB1-ETL-pipeline.ipynb`**
*(Raw Data ETL & Anomaly Flagging for 10 Routers)*

**Purpose:** Loads the individual router Parquet files (from the Prerequisite Notebook), performs Dask-native cleaning and hourly aggregation, conducts Isolation Forest anomaly detection, and saves processed hourly files for each router. Generates initial EDA plots.

1.  Open `NB1-ETL-pipeline.ipynb`.
2.  Verify `INPUT_ROUTERS_DIR` in `Configuration`.
3.  Run all cells sequentially.

**Expected Output:** Console logs detailing processing per router, 10 `hourly_<router>_processed_with_anomalies.parquet` files, and initial EDA figures (Router Inventory Table, Daily Utilization Curves, Flow-size Distribution, Weekday Heatmap, Anomaly Plots) in `outputs/ch4/figs/`.

---

### **Step 2: Execute `NB2-CPU-Models-without-patchTST.ipynb`**
*(CPU-Friendly Models Training & Evaluation for 10 Routers)*

**Purpose:** Loads the processed hourly data (from Step 1) for all routers. For each router, it splits/scales data and trains/evaluates **GRU-LSTM, SARIMA, NBEATS, and TiDE** across their supported anomaly integration strategies.

1.  Open `NB2-CPU-Models-without-patchTST.ipynb`.
2.  Verify `INPUT_PROCESSED_HOURLY_DIR` in `Configuration`.
3.  Run all cells sequentially.

**Expected Output:** Console logs with metrics for each run. Numerous small Parquet files saved to `outputs/ch4/predictions/` (`y_true`, `y_pred` for test sets). An `outputs/ch4/all_model_results.csv` file created/updated with consolidated metrics.

---

### **Step 3: Execute `NB3-GPU-models-patchTST.ipynb`**
*(Advanced Models & Final Visualizations for 10 Routers)*

**Purpose:** Completes the model suite by training/evaluating **PatchTST** and a **DCRNN placeholder**. It then loads the full collected results and generates final benchmark and conclusion visualizations.

1.  Open `NB3-GPU-models-patchTST.ipynb`.
2.  Verify `INPUT_PROCESSED_HOURLY_DIR` in `Configuration`.
3.  Run all cells sequentially.

**Expected Output:** Console logs for PatchTST and DCRNN runs. `all_model_results.csv` updated. Final benchmark figures (Computation Time, Performance Radar Chart) saved to `outputs/ch4/figs/`. A final summary printed.

---

## Technical Challenges Overcome

The pipeline's robust design is a result of overcoming several significant technical hurdles:
*   **Memory Management for Large Raw Data:** Efficiently processing 300+ GB of raw Netflow data within 60 GB RAM by leveraging Dask for lazy loading, aggressive column pruning, Dask-native hourly aggregation, and only computing small, processed data portions into Pandas. This was particularly crucial for the initial conversion into individual Parquet files and subsequent hourly aggregation.
*   **Data Inconsistency:** Addressing potential row loss during initial large-scale data transformations by implementing explicit data integrity checks and refined Dask workflows.
*   **Complex API Integrations:** Meticulously debugging and aligning with the precise API nuances of `NeuralForecast` models and `pytorch_lightning` callbacks for multi-horizon, anomaly-aware forecasting.
*   **Modularization:** Breaking down a monolithic workflow into distinct, manageable Jupyter Notebooks for enhanced debuggability and a clear, sequential execution path.

This robust framework now provides a solid foundation for further performance tuning and comprehensive analysis.
