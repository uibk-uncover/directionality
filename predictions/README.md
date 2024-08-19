This folder provides the predictions and ground truth labels for the CNN-based experiments. These allow you to reconstruct Table 3, Table 6, and Table 8 from the paper.
Unfortunately, the prediction vectors for the DALL-E mini detector are not included. We had already removed the corresponding images to make space for more experiments. 

Example how to reconstruct the steganalysis results (Table 3 in the paper):

```python
from pathlib import Path
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import re


predictions_base_dir = "./steganalysis"
predictions_csv_files = list(glob(os.path.join(predictions_base_dir, "**", "*.csv.gz"), recursive=True))


def parse_csv_file(csv_file):
    df = pd.read_csv(csv_file, compression="gzip")

    # Sanity checks
    df["basename"] = df["filename"].map(os.path.basename)
    cover_mask = df["y_true"] == 0
    stego_mask = df["y_true"] == 1
    
    # Are classes in the test set balanced?
    assert np.sum(cover_mask) == np.sum(stego_mask)
    
    # Check filenames
    assert set(df[cover_mask]["basename"]) == set(df[stego_mask]["basename"])
    
    # Compute accuracy
    accuracy = np.mean(df["y_pred"] == df["y_true"])
    
    # Extract experimental setup and test orientation from the file path
    path = Path(os.path.relpath(csv_file, predictions_base_dir))
    match = re.search("setup_([a-z]+)_orientation_([a-z0-9]+)", path.stem)
    
    # Return results
    return {
        "stego_method": path.parts[0],
        "qt": path.parts[1],
        "setup": match.group(1),
        "orientation": match.group(2),
        "accuracy": accuracy,
    }


# Read all csv files
buffer = []

for csv_file in tqdm(predictions_csv_files):
    res_dict = parse_csv_file(csv_file)
    buffer.append(res_dict)
    
df = pd.DataFrame(buffer)

# Convert to categorical for sorting
df["stego_method"] = pd.Categorical(df["stego_method"], ["nsF5", "UERD", "J-UNIWARD"])
df["setup"] = pd.Categorical(df["setup"], ["norot", "augrot", "baserot"])
df["qt"] = pd.Categorical(df["qt"], ["standard-qt-75", "asymmetric-qt-80-60"])

# Reproduce Table 3 from the paper
df.sort_values(["stego_method", "qt", "setup", "orientation"]).pivot(
    index=["setup", "stego_method"],
    columns=["qt", "orientation"],
    values=["accuracy"],
).round(3)
```