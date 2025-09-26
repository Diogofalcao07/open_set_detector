import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Font and display settings for visibility
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold"
})

# === Load evaluation JSON files ===
with open("evaluation_results_detectron.json", "r") as f:
    detectron_data = json.load(f)

with open("test_open_question_detector.json", "r") as f:
    chatrex_data = json.load(f)

# === Convert to DataFrames ===
df_detectron = pd.DataFrame(detectron_data)
df_detectron.rename(columns={"Average Precision": "AP", "Average Recall": "AR"}, inplace=True)
df_detectron["Model"] = "Detectron2"

df_chatrex = pd.DataFrame(chatrex_data)
df_chatrex.rename(columns={"Average Precision": "AP", "Average Recall": "AR"}, inplace=True)
df_chatrex["Model"] = "ChatREX"

# === Combine and clean ===
combined = pd.concat([df_detectron, df_chatrex])
combined = combined.dropna(subset=["AP", "AR"])

# Keep only categories present in both models
common_categories = set(df_detectron["Category"]).intersection(df_chatrex["Category"])
combined = combined[combined["Category"].isin(common_categories)]

# Filter by AP/AR threshold
threshold = 0
filtered_categories = combined[
    (combined["AP"] > threshold) | (combined["AR"] > threshold)
]["Category"].unique()
combined = combined[combined["Category"].isin(filtered_categories)]

# Sort categories by Detectron2 AP
detectron_sorted = combined[combined["Model"] == "Detectron2"]
category_order = detectron_sorted.sort_values(by="AP", ascending=True)["Category"]

# === Plotting ===
y = np.arange(len(category_order))
bar_height = 0.35
fig_height = len(y) * 0.6 + 6
fig, axs = plt.subplots(1, 2, figsize=(18, fig_height), sharey=True)

colors = {
    "Detectron2": "#1f77b4",
    "ChatREX": "#ff7f0e"
}

for i, metric in enumerate(["AP", "AR"]):
    detectron_vals = combined[combined["Model"] == "Detectron2"].set_index("Category").loc[category_order][metric]
    chatrex_vals = combined[combined["Model"] == "ChatREX"].set_index("Category").loc[category_order][metric]

    axs[i].barh(y - bar_height/2, detectron_vals, bar_height, label="Detectron2", color=colors["Detectron2"])
    axs[i].barh(y + bar_height/2, chatrex_vals, bar_height, label="ChatREX", color=colors["ChatREX"])
    axs[i].axvline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5' if i == 0 else None)

    axs[i].set_xlabel(metric, fontweight='bold')
    axs[i].set_title(f"{metric} Comparison by Category", fontweight='bold')
    axs[i].legend()
    axs[i].grid(True, axis='x')

# Add category names to y-axis
axs[0].set_yticks(y)
axs[0].set_yticklabels(category_order, fontweight='bold')

plt.tight_layout(pad=3.0)
plt.savefig("detectron_vs_chatrex_comparison.pdf")
plt.show()
