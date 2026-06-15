# Functional Document: AI Parkinson Detection System

## 1. Introduction
The AI Parkinson Detection System project aims to provide an accessible, non-invasive screening tool for Parkinson’s disease using voice analysis. By leveraging a machine learning model integrated into a web dashboard, the system allows for rapid assessment of patient voice recordings to detect symptoms associated with neurodegeneration.

## 2. Product Goal
The primary goal of this product is to deliver a reliable, real-time prediction tool that assists clinicians and users in assessing Parkinson's risk based on vocal biomarkers (Jitter, Shimmer, and PPE). This aligns with the overarching objective of enabling early detection and continuous monitoring of neurological health through accessible web technology.

## 3. Demography (Users, Location)
**Users**
* Target Users: Clinicians, medical researchers, and self-screening patients.
* User Characteristics: Varying levels of medical expertise and technical proficiency.

**Location**
* Target Location: Global access via web browser, specifically targeting clinics or home environments with microphone access.

## 4. Business Processes
The key business processes include:
* **Patient Data Entry:** Process for users to input patient identification and assessment dates.
* **Vocal Assessment (Manual & Audio):** Process for users to either manually input known acoustic metrics or record a live voice sample for automated metric extraction.
* **Risk Analysis and Reporting:** Process for the system to evaluate metrics against the AI model, generate a risk probability score, and compile a medical PDF report.

## 5. Features
This system focuses on implementing the following key features:

### 5.1 Manual Data Prediction
1. **Description:** Allows clinicians who already have vocal metrics (Jitter, Shimmer, PPE) to manually input them into the system for an instant prediction and risk level assessment.
2. **User Story:** As a clinician, I want to manually input acoustic metrics obtained from my own software so that I can use the AI model to quickly determine a patient's Parkinson's risk level.

### 5.2 Live Voice Recording and Analysis
1. **Description:** Integrates a built-in audio recorder that captures 5 seconds of the user's voice, automatically extracts the necessary acoustic features using librosa, and runs the prediction.
2. **User Story:** As a self-screening patient or clinician without specialized audio software, I want to record my voice directly into the app so that the system can automatically extract my acoustic features and provide a diagnosis.

### 5.3 Automated Medical Reporting
1. **Description:** Generates a downloadable PDF report detailing the patient's name, test date, classification result, and a visual graph of the acoustic features.
2. **User Story:** As a healthcare provider, I want to download a formatted PDF report of the assessment so that I can attach it to the patient's electronic health record.

## 6. Authorization Matrix

| Role | Access Level |
| :--- | :--- |
| **Clinician / User** | Full access to input data, record voice, view predictions, and download PDF reports. |
| **Admin** | Access to backend scripts, environment configuration, and model retraining (model.py). |

*(Note: Currently, the frontend dashboard is open access without an authentication layer.)*

## 7. Assumptions
* The host machine running the web application possesses the necessary audio recording hardware (microphone) and drivers to utilize `sounddevice`.
* Users consent to providing their voice samples for immediate processing.
* The development environment and infrastructure (Python, Streamlit, XGBoost) remain stable.
* The pre-trained XGBoost model ([model.pkl](file:///c:/Users/HP/Desktop/Neural-Degradation-Project/model.pkl)) maintains its 90%+ accuracy when exposed to new patient audio data.
