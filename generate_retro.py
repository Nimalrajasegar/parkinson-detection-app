import pandas as pd

retro_data = [
    {
        "Category": "What Went Well",
        "Item": "Model Training",
        "Description": "Successfully trained an XGBoost model with data scaling (StandardScaler) using the core voice features (Jitter, Shimmer, PPE).",
        "Owner": "Data Science Team"
    },
    {
        "Category": "What Went Well",
        "Item": "Frontend Implementation",
        "Description": "Built a responsive and interactive web interface using Streamlit, facilitating both manual and voice inputs.",
        "Owner": "Frontend Developer"
    },
    {
        "Category": "What Went Well",
        "Item": "Medical Reporting",
        "Description": "Implemented automated PDF report generation using reportlab, including risk level and plotting data.",
        "Owner": "Backend Developer"
    },
    {
        "Category": "What Could Be Improved",
        "Item": "Voice Recording in Web",
        "Description": "Current use of 'sounddevice' for audio recording might not work in cloud/web deployments. It expects local audio hardware.",
        "Owner": "Architecture Team"
    },
    {
        "Category": "What Could Be Improved",
        "Item": "Feature Engineering",
        "Description": "Currently relying on only 3 features. We have more features in data.csv that could improve model robustness.",
        "Owner": "Data Science Team"
    },
    {
        "Category": "What Could Be Improved",
        "Item": "UI/UX Polish",
        "Description": "The web application looks okay but can be improved beyond basic Streamlit theming.",
        "Owner": "Frontend Developer"
    },
    {
        "Category": "Action Item",
        "Item": "Refactor Audio Input",
        "Description": "Investigate Streamlit-native web audio recorder components (like streamlit-webrtc or audio-recorder-streamlit) to replace sounddevice.",
        "Owner": "Architecture Team"
    },
    {
        "Category": "Action Item",
        "Item": "Model Evaluation",
        "Description": "Conduct a new model training experiment incorporating more features from data.csv.",
        "Owner": "Data Science Team"
    },
    {
        "Category": "Action Item",
        "Item": "Error Handling",
        "Description": "Add robust try-except blocks, especially around audio processing and librosa feature extraction.",
        "Owner": "Backend Developer"
    }
]

df = pd.DataFrame(retro_data)
df.to_csv("sprint_retrospective.csv", index=False)
print("Sprint retrospective successfully saved to sprint_retrospective.csv")
