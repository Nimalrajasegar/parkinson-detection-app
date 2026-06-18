import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pandas as pd
from datetime import datetime

tasks = [
    {"Task": "Require & Design (FSD)", "Start": "2024-04-01", "Duration": 3, "Color": "#1f77b4"},
    {"Task": "Data Prep & ML Model", "Start": "2024-04-03", "Duration": 5, "Color": "#ff7f0e"},
    {"Task": "Frontend (Streamlit)", "Start": "2024-04-06", "Duration": 4, "Color": "#2ca02c"},
    {"Task": "Voice / Audio Module", "Start": "2024-04-08", "Duration": 4, "Color": "#d62728"},
    {"Task": "PDF Medical Reports", "Start": "2024-04-10", "Duration": 3, "Color": "#9467bd"},
    {"Task": "Testing & QA", "Start": "2024-04-12", "Duration": 3, "Color": "#8c564b"},
    {"Task": "Deployment", "Start": "2024-04-14", "Duration": 2, "Color": "#e377c2"}
]

df = pd.DataFrame(tasks)
df['Start'] = pd.to_datetime(df['Start'])
df['End'] = df['Start'] + pd.to_timedelta(df['Duration'], unit='D')

fig, ax = plt.subplots(figsize=(10, 6))


for i, task in enumerate(reversed(df.to_dict('records'))):
    start = task['Start']
    duration = task['Duration']
    ax.barh(task['Task'], duration, left=start, color=task['Color'], height=0.5, align='center', edgecolor='black')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))

plt.title("Parkinson Detection Project - Work Plan & Timeline", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Project Timeline (April 2024)", fontsize=12, labelpad=10)

plt.grid(axis='x', linestyle='--', alpha=0.6)


plt.tight_layout()

plt.savefig("work_plan.png", dpi=300, facecolor='white', bbox_inches='tight')
print("Successfully generated work_plan.png")
