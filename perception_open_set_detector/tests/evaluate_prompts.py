import json, os
from pycocotools.coco import COCO

from perception_open_set_detector.utils.core_functions import create_coco_evaluator, extract_category_metrics

# Prepare GroundTruth data for COCOeval()
ground_truth_path = "perception_open_set_detector/coco_data/merged_annotations.coco.json"
coco_gt = COCO(ground_truth_path)

# === Detection files ===
detection_files = sorted([f for f in os.listdir() if f.startswith("test_prompt_") and f.endswith(".json")])
print(f"Found detection files: {detection_files}")

all_results = []

# === Evaluate each prompt ===
for i, det_file in enumerate(detection_files):

    with open(det_file, "r") as f:
        all_detections = json.load(f)

    if not all_detections:
        print(f"⚠️ Skipping empty: {det_file}")
        continue

    coco_dt = coco_gt.loadRes(all_detections)
    coco_eval = create_coco_evaluator(coco_gt, coco_dt, all_detections)
    
    print(f"\nResults for {prompt_labels[i]}: {prompts[i]}")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    precisions = coco_eval.eval['precision']
    categories = coco_gt.loadCats(coco_eval.params.catIds)
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}

    prompt_results = []
    for idx, catId in enumerate(coco_eval.params.catIds):
        precision = precisions[:, :, idx, 0, 2]
        precision = precision[precision > -1]
        ap = float('nan') if precision.size == 0 else np.mean(precision)
        prompt_results.append({
            "Category": cat_id_to_name[catId],
            "AP": ap
        })
    all_results.append({
        "prompt_label": prompt_labels[i],
        "prompt_text": prompts[i],
        "results": prompt_results
    })

# === Combine all into DataFrame ===
dfs = []
for r in all_results:
    df = pd.DataFrame(r["results"]).set_index("Category")
    df = df.rename(columns={"AP": r["prompt_label"]})
    dfs.append(df)
df_all = pd.concat(dfs, axis=1)
df_all = df_all.sort_index()

# === Filter categories with high AP std deviation (>0.1) ===
ap_std = df_all.std(axis=1)
distinctive_cats = ap_std[ap_std > 0.1].index.tolist()
print(f"\nDistinctive categories (std > 0.1): {distinctive_cats}")

df_distinct = df_all.loc[distinctive_cats]

# === Plotting vertically stacked with categories on Y-axis ===
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(8, 18))  # Narrower width (8), keep tall height (18)
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 4, 4], hspace=0.4)

# Legend subplot (top)
ax_legend = fig.add_subplot(gs[0])
ax_legend.axis('off')
legend_text = "\n".join([f"{label}: {text}" for label, text in zip(prompt_labels, prompts)])
ax_legend.text(0, 0.5, "Prompt Legend:\n" + legend_text, fontsize=11, va='center', ha='left')

# Horizontal bar chart subplot (middle)
ax_bar = fig.add_subplot(gs[1])
df_distinct.plot(kind='barh', ax=ax_bar)
ax_bar.set_title("AP by Prompt for Distinctive Categories")
ax_bar.set_xlabel("Average Precision (AP)")
ax_bar.set_ylabel("Category")
ax_bar.legend(title="Prompt", bbox_to_anchor=(1.05, 1), loc='upper left')
ax_bar.grid(axis='x')

# Heatmap subplot (bottom) — make narrower by reducing width of axis
ax_heat = fig.add_subplot(gs[2])
sns.heatmap(df_distinct, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'AP'}, ax=ax_heat)

ax_heat.set_title("Heatmap of AP Scores for Distinctive Categories")
ax_heat.set_ylabel("Category")
ax_heat.set_xlabel("Prompt")

plt.tight_layout()
plt.savefig("comparison_distinctive_categories_vertical_yaxis.pdf")
plt.show()


print("\n✅ Saved PDF: comparison_distinctive_categories_vertical_yaxis.pdf")




