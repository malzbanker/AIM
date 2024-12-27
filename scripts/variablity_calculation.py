import numpy as np

def calculate_variability(data):
    for feature, stats in data.items():
        # Extract statistics
        mean = stats["mean"]
        std = stats["std"]
        min_val = stats["min"]
        max_val = stats["max"]
        q1 = stats["25%"]
        q3 = stats["75%"]
        
        # Variability measures
        variance = std ** 2
        cv = (std / mean) * 100 if mean != 0 else np.nan  # Handle divide by zero
        range_val = max_val - min_val
        iqr = q3 - q1
        
        # Print results
        print(f"{feature} Variability:")
        print(f"  Variance: {variance:.2f}")
        print(f"  Coefficient of Variation: {cv:.2f}%")
        print(f"  Range: {range_val:.2f}")
        print(f"  IQR: {iqr:.2f}\n")

# Calculate and display variability
calculate_variability(data[['TotalPremium', 'TotalClaims']].describe())