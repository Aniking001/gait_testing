import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter
import os

# File paths
DATA_DIR = r"D:\SEM 6\gait\data"
FEMALE_FILE = os.path.join(DATA_DIR, "female_gait_angles.csv")
MALE_FILE = os.path.join(DATA_DIR, "male_gait_angles.csv")
CLEANED_OUTPUT_DIR = DATA_DIR

def load_and_inspect_data(file_path, gender):
    """Load data and show basic statistics"""
    df = pd.read_csv(file_path)
    print(f"\n=== {gender.upper()} DATA INSPECTION ===")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Basic stats:\n{df.describe()}")
    return df

def clean_gait_data(df, gender):
    """Clean the gait angle data"""
    df_clean = df.copy()
    
    # Remove rows where all angle values are NaN
    angle_cols = ['hip_angle_r', 'knee_angle_r', 'foot_strike_angle_r', 
                  'hip_angle_l', 'knee_angle_l', 'foot_strike_angle_l', 'torso_angle']
    
    # Remove frames with no pose detection
    df_clean = df_clean.dropna(subset=angle_cols, how='all')
    
    # Handle outliers using IQR method
    for col in angle_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing (better for time series)
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
    
    # Smooth the data using Savitzky-Golay filter
    for col in angle_cols:
        if col in df_clean.columns and len(df_clean) > 10:
            # Fill NaN with interpolation first
            df_clean[col] = df_clean[col].interpolate(method='linear')
            # Apply smoothing
            if not df_clean[col].isna().all():
                try:
                    df_clean[col + '_smooth'] = savgol_filter(df_clean[col].fillna(method='ffill'), 
                                                            window_length=min(11, len(df_clean)//3), 
                                                            polyorder=2)
                except:
                    df_clean[col + '_smooth'] = df_clean[col]
    
    # Add derived features
    df_clean['stride_phase'] = df_clean.index % 60  # Assuming ~60 frames per stride
    df_clean['hip_symmetry'] = abs(df_clean['hip_angle_r'] - df_clean['hip_angle_l'])
    df_clean['knee_symmetry'] = abs(df_clean['knee_angle_r'] - df_clean['knee_angle_l'])
    
    print(f"\n=== {gender.upper()} DATA CLEANING RESULTS ===")
    print(f"Original frames: {len(df)}")
    print(f"Cleaned frames: {len(df_clean)}")
    print(f"Frames removed: {len(df) - len(df_clean)}")
    
    return df_clean

def visualize_gait_patterns(df_female, df_male):
    """Create comprehensive visualizations of gait patterns"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    
    # 1. Angle Distribution Comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('Gait Angle Distributions: Male vs Female', fontsize=16, fontweight='bold')
    
    angle_cols = ['hip_angle_r', 'knee_angle_r', 'foot_strike_angle_r', 
                  'hip_angle_l', 'knee_angle_l', 'foot_strike_angle_l', 'torso_angle']
    
    for i, col in enumerate(angle_cols):
        row = i // 4
        col_idx = i % 4
        if col_idx < 4:
            axes[row, col_idx].hist(df_female[col].dropna(), alpha=0.6, label='Female', bins=30, color='pink')
            axes[row, col_idx].hist(df_male[col].dropna(), alpha=0.6, label='Male', bins=30, color='lightblue')
            axes[row, col_idx].set_title(f'{col.replace("_", " ").title()}')
            axes[row, col_idx].legend()
            axes[row, col_idx].set_xlabel('Angle (degrees)')
            axes[row, col_idx].set_ylabel('Frequency')
    
    # Remove empty subplots
    for i in range(len(angle_cols), 8):
        row = i // 4
        col_idx = i % 4
        axes[row, col_idx].remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CLEANED_OUTPUT_DIR, 'gait_angle_distributions.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Time Series Analysis
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Gait Angle Time Series (First 500 frames)', fontsize=16, fontweight='bold')
    
    sample_frames = min(500, len(df_female), len(df_male))
    
    for i, col in enumerate(angle_cols):
        row = i // 3
        col_idx = i % 3
        axes[row, col_idx].plot(df_female[col][:sample_frames], label='Female', alpha=0.8, color='red')
        axes[row, col_idx].plot(df_male[col][:sample_frames], label='Male', alpha=0.8, color='blue')
        axes[row, col_idx].set_title(f'{col.replace("_", " ").title()}')
        axes[row, col_idx].legend()
        axes[row, col_idx].set_xlabel('Frame')
        axes[row, col_idx].set_ylabel('Angle (degrees)')
        axes[row, col_idx].grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(angle_cols), 9):
        row = i // 3
        col_idx = i % 3
        axes[row, col_idx].remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CLEANED_OUTPUT_DIR, 'gait_time_series.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Correlation Heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Female correlations
    corr_female = df_female[angle_cols].corr()
    sns.heatmap(corr_female, annot=True, cmap='coolwarm', center=0, ax=ax1)
    ax1.set_title('Female Gait Angle Correlations')
    
    # Male correlations
    corr_male = df_male[angle_cols].corr()
    sns.heatmap(corr_male, annot=True, cmap='coolwarm', center=0, ax=ax2)
    ax2.set_title('Male Gait Angle Correlations')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CLEANED_OUTPUT_DIR, 'gait_correlations.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Statistical Summary
    print("\n=== STATISTICAL SUMMARY ===")
    print("\nFEMALE STATISTICS:")
    print(df_female[angle_cols].describe())
    print("\nMALE STATISTICS:")
    print(df_male[angle_cols].describe())

def save_cleaned_data(df_female, df_male):
    """Save cleaned datasets"""
    female_output = os.path.join(CLEANED_OUTPUT_DIR, "female_gait_cleaned.csv")
    male_output = os.path.join(CLEANED_OUTPUT_DIR, "male_gait_cleaned.csv")
    
    df_female.to_csv(female_output, index=False)
    df_male.to_csv(male_output, index=False)
    
    print(f"\n=== DATA SAVED ===")
    print(f"Female cleaned data: {female_output}")
    print(f"Male cleaned data: {male_output}")
    
    # Create combined dataset for model training
    df_female['gender'] = 'female'
    df_male['gender'] = 'male'
    combined = pd.concat([df_female, df_male], ignore_index=True)
    combined_output = os.path.join(CLEANED_OUTPUT_DIR, "combined_gait_cleaned.csv")
    combined.to_csv(combined_output, index=False)
    print(f"Combined dataset: {combined_output}")
    
    return female_output, male_output, combined_output

def main():
    """Main execution function"""
    print("=== GAIT DATA CLEANING AND ANALYSIS ===")
    
    # Load data
    df_female_raw = load_and_inspect_data(FEMALE_FILE, "female")
    df_male_raw = load_and_inspect_data(MALE_FILE, "male")
    
    # Clean data
    df_female_clean = clean_gait_data(df_female_raw, "female")
    df_male_clean = clean_gait_data(df_male_raw, "male")
    
    # Visualize patterns
    visualize_gait_patterns(df_female_clean, df_male_clean)
    
    # Save cleaned data
    female_file, male_file, combined_file = save_cleaned_data(df_female_clean, df_male_clean)
    
    print(f"\n=== PROCESS COMPLETE ===")
    print("Your cleaned datasets are ready for model training!")
    print("Check the generated visualizations to understand correct running patterns.")

if __name__ == "__main__":
    main()
