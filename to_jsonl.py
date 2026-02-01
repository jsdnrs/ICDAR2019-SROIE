import os
import json
import glob

from PIL import Image

def get_identifier_from_path(path):
  file_name = os.path.basename(path)
  (identifier, _) = os.path.splitext(file_name)
  return identifier

def load_images(prefix):
  data = {}
  for path in glob.iglob(os.path.join(prefix, "images", "*.jpg")):
    name = get_identifier_from_path(path)

    data[name] = {}
    data[name]["key"] = name

    data[name]["image_size"] = {}
    with Image.open(path) as img:
      (width, height) = img.size
      data[name]["image_size"]["width"] = width
      data[name]["image_size"]["height"] = height
  
  return data


def load_task1_2(prefix, data):
  for path in glob.iglob(os.path.join(prefix, "task1_2", "*.txt")):
    name = get_identifier_from_path(path)

    with open(path, mode="r") as f:
      data[name]["words"] = []
      data[name]["bboxes"] = []
      for line in f.readlines():
        tokens = line.strip().split(",")
        if len(tokens) == 1: # Skip empty lines
          continue
        data[name]["words"].append(tokens[8])
        
        (x1, y1, x2, y2, x3, y3, x4, y4) = map(int, tokens[:8])

        # COCO-format 
        # data[name]["bboxes"].append([
        #   x1, y1, (x3-x1), (y3-y1)])
        
        data[name]["bboxes"].append([
          x1, y1, x3, y3])


def load_task3(prefix, data):
  for path in glob.iglob(os.path.join(prefix, "task3", "*.txt")):
    name = get_identifier_from_path(path)
    
    with open(path, mode="r") as f:
      data[name]["entities"] = json.load(f)


def main(split):
  prefix = os.path.join("orig", split)
  data = load_images(prefix)

  load_task3(prefix, data)
  if os.path.isdir(os.path.join("additional_data", split)):
    load_task3(os.path.join("additional_data", split), data)

  load_task1_2(prefix, data)
  
  out = os.path.join("data", split, f"metadata.jsonl")
  # out = os.path.join("data", f"{split}.jsonl")
  with open(out, mode="w") as f:
    for file_name, values in data.items():
      f.write(json.dumps({"file_name": f"{file_name}.jpg", **values}) + '\n')


if __name__ == "__main__":
  main("train")
  main("test")
