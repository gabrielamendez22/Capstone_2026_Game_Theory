import json
import csv

# Change this to your actual results filename
input_file  = "pd_results_20260327_112138.json"
output_file = "pd_results_20260327_100443.csv"

with open(input_file, "r") as f:
    data = json.load(f)

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

print(f"✅ Saved {len(data)} rows to {output_file}")