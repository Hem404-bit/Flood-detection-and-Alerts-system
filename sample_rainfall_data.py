"""
Generate sample rainfall dataset for Rainfall Analysis module.
Run this once to create rainfall_dataset.csv
"""

import pandas as pd
import numpy as np

def generate_rainfall_data():
    subdivisions = [
        "ANDAMAN & NICOBAR ISLANDS", "ARUNACHAL PRADESH", "ASSAM & MEGHALAYA",
        "BIHAR", "CHHATTISGARH", "COASTAL ANDHRA PRADESH", "COASTAL KARNATAKA",
        "EAST MADHYA PRADESH", "EAST RAJASTHAN", "EAST UTTAR PRADESH",
        "GANGETIC WEST BENGAL", "GUJARAT REGION", "HARYANA DELHI & CHANDIGARH",
        "HIMACHAL PRADESH", "JAMMU & KASHMIR", "JHARKHAND", "KERALA",
        "KONKAN & GOA", "LAKSHADWEEP", "MARATHWADA", "NORTH INTERIOR KARNATAKA",
        "ORISSA", "PUNJAB", "RAYALASEEMA", "SAURASHTRA & KUTCH", "SIKKIM",
        "SOUTH INTERIOR KARNATAKA", "SUB HIMALAYAN WEST BENGAL & SIKKIM",
        "TAMIL NADU", "TELANGANA", "UTTARAKHAND", "VIDARBHA",
        "WEST MADHYA PRADESH", "WEST RAJASTHAN", "WEST UTTAR PRADESH",
    ]
    
    monsoon_peaks = {
        "KERALA": [0,0,0.5,1,3,18,25,22,14,8,3,0.5],
        "ASSAM & MEGHALAYA": [0.5,1,3,8,14,18,22,20,14,6,1,0.5],
        "RAJASTHAN": [0,0,0,0,0.5,3,18,22,5,1,0,0],
        "TAMIL NADU": [1,0.5,0.3,0.5,1,3,8,12,14,18,12,5],
    }
    
    rows = []
    rng = np.random.default_rng(42)
    
    for sub in subdivisions:
        # Select pattern based on substring match
        key = next((k for k in monsoon_peaks if k in sub), None)
        if key:
            pattern = np.array(monsoon_peaks[key])
        else:
            # Default monsoon pattern
            pattern = np.array([0.3,0.5,1,2,4,14,22,20,12,5,1,0.4])
        
        base_annual = rng.uniform(600, 2800)
        
        for yr in range(1901, 2020):
            # Add interannual variability
            noise = rng.normal(1.0, 0.18)
            months = pattern / (pattern.sum() + 1e-6) * base_annual * max(0.3, noise)
            months = np.abs(months + rng.normal(0, 5, 12))
            
            rows.append({
                "SUBDIVISION": sub,
                "YEAR": yr,
                "JAN": months[0], "FEB": months[1], "MAR": months[2],
                "APR": months[3], "MAY": months[4], "JUN": months[5],
                "JUL": months[6], "AUG": months[7], "SEP": months[8],
                "OCT": months[9], "NOV": months[10], "DEC": months[11],
                "ANNUAL": months.sum(),
            })
    
    df = pd.DataFrame(rows)
    df.to_csv("rainfall_dataset.csv", index=False)
    print(f"✓ Generated rainfall_dataset.csv with {len(df)} records")
    print(f"  Subdivisions: {len(subdivisions)}")
    print(f"  Years: 1901–2019")

if __name__ == "__main__":
    generate_rainfall_data()
