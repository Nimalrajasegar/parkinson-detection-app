# Project Architecture Analysis

Based on the source code in your directory, here is an analysis of the Neural-Degradation-Project.

## Overview
The project is a **Parkinson's Disease Detection System** built with Python. It uses an **XGBoost** machine learning model to classify whether a patient has Parkinson's based on vocal features (Jitter, Shimmer, and PPE). It provides a user-friendly frontend using **Streamlit** where users can either manually input these features or record their voice directly to get a real-time prediction and generate a medical report.

## Components Breakdown

1. **Data & Training Pipeline ([model.py](file:///c:/Users/HP/Desktop/Neural-Degradation-Project/model.py), [main.py](file:///c:/Users/HP/Desktop/Neural-Degradation-Project/main.py))**
   - **Data Input:** Loads patient voice metrics from [data.csv](file:///c:/Users/HP/Desktop/Neural-Degradation-Project/data.csv).
   - **Preprocessing:** Extracts three core features: `MDVP:Jitter(%)`, `MDVP:Shimmer`, and `PPE`. It uses `StandardScaler` to normalize the data.
   - **Model:** Trains an `XGBClassifier` (XGBoost) model.
   - **Artifacts:** Exports the trained model to [model.pkl](file:///c:/Users/HP/Desktop/Neural-Degradation-Project/model.pkl) and the fitted scaler to [scaler.pkl](file:///c:/Users/HP/Desktop/Neural-Degradation-Project/scaler.pkl) to be consumed by the web app.

2. **Web Application ([app.py](file:///c:/Users/HP/Desktop/Neural-Degradation-Project/app.py))**
   - **Framework:** Built with Streamlit for a responsive, interactive UI.
   - **Input Modalities:**
     - **Manual:** Clinicians can manually enter Jitter, Shimmer, and PPE metrics.
     - **Voice (Audio):** Uses `sounddevice` to record a 5-second audio clip ([voice.wav](file:///c:/Users/HP/Desktop/Neural-Degradation-Project/voice.wav)), then uses the `librosa` library to dynamically extract the three acoustic features from the audio.
   - **Prediction & Feedback:** Loads the [.pkl](file:///c:/Users/HP/Desktop/Neural-Degradation-Project/model.pkl) files, scales inputs, and predicts the likelihood of Parkinson's (`No Parkinson` / `Parkinson`), showing real-time probability and Risk Level (Low/Medium/High).
   - **Reporting:** Uses `matplotlib` to plot data distribution or audio waveforms, and `reportlab` to generate a downloadable PDF Medical Report.

## Architecture Diagram

```mermaid
graph TD
    classDef file fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#bbf,stroke:#333,stroke-width:2px;
    classDef data fill:#dfd,stroke:#333,stroke-width:2px;
    classDef ui fill:#fdd,stroke:#333,stroke-width:2px;

    subgraph Training Pipeline
        A[data.csv]:::data --> B(model.py / main.py):::process
        B --> C[StandardScaler]:::process
        B --> D[XGBoost Classifier]:::process
        C --> E[(scaler.pkl)]:::file
        D --> F[(model.pkl)]:::file
    end

    subgraph Web UI Streamlit
        G[Manual Input]:::ui
        H[Voice Input]:::ui
        H -->|Records 5s| I(voice.wav):::file
        I -->|librosa Extraction| J(Feature Extraction)
    end

    subgraph Backend Application app.py
        K[Load Models & Scale]:::process
        L[Prediction Engine]:::process
        M[Generate PDF / Graphs]:::process
        
        E --> K
        F --> L
        G --> K
        J --> K
        K --> L
        L --> M
    end

    M --> N[UI Display: Results & Risk Level]:::ui
    M --> O[Downloadable report.pdf]:::data
```
