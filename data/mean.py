import pandas as pd

# Load your correct posture dataset (CSV or Excel)
df = pd.read_csv("D:\\SEM 6\\gait\\data\\male_gait_cleaned.csv")  # or .xlsx

angle_cols = ['hip_angle_r', 'knee_angle_r', 'foot_strike_angle_r',
              'hip_angle_l', 'knee_angle_l', 'foot_strike_angle_l', 'torso_angle']

ranges = {}
for col in angle_cols:
    mean = df[col].mean()
    std = df[col].std()
    lower = mean - 2*std
    upper = mean + 2*std
    ranges[col] = (lower, upper)
    print(f"{col}: {lower:.2f} to {upper:.2f}")

# Now 'ranges' holds the correct posture ranges for each angle.
