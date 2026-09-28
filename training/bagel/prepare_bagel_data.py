"""Convert an ms-swift training JSONL into BAGEL's llava-style VLM SFT format.

    python prepare_bagel_data.py <task> <train.jsonl> <train_dir> <work_dir>

Writes <work_dir>/<task>_bagel.jsonl, dataset_info.json (registered through
BAGEL_EXTRA_VLM_SFT, see bagel.patch) and dataset_config.yaml.
"""
import json
import os
import sys

task, jsonl, train_dir, work = sys.argv[1:5]
rows, image_dir = [], None
for line in open(jsonl):
    d = json.loads(line)
    user = next(m for m in d["messages"] if m["role"] == "user")
    answer = next(m for m in d["messages"] if m["role"] == "assistant")
    path = d["images"][0].replace("__DATA_ROOT__", train_dir)
    image_dir = image_dir or os.path.dirname(path)
    if os.path.dirname(path) != image_dir:
        raise SystemExit("all images of a task must share one directory")
    rows.append({"image": os.path.basename(path), "conversations": [
        {"from": "human", "value": "<image>\n" + user["content"].replace("<image>", "", 1).strip()},
        {"from": "gpt", "value": answer["content"]}]})
out = os.path.join(work, f"{task}_bagel.jsonl")
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
with open(os.path.join(work, "dataset_info.json"), "w") as f:
    json.dump({task: {"data_dir": image_dir, "jsonl_path": out, "num_total_samples": len(rows)}}, f, indent=1)
with open(os.path.join(work, "dataset_config.yaml"), "w") as f:
    f.write(f"""vlm_sft:
  dataset_names:
  - {task}
  image_transform_args:
    image_stride: 14
    max_image_size: 490
    min_image_size: 252
  is_mandatory: true
  num_used_data:
  - {len(rows)}
  weight: 1
""")
print(f"{task}: {len(rows)} rows, images in {image_dir}")
