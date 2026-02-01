import os
import glob
import json

from webdataset import TarWriter

def to_webdataset(split, out, prefix="."):
  path = os.path.join(prefix, f"{split}.jsonl")
  with TarWriter(out, encoder=False) as sink:
    with open(path, mode="r") as f:
      for line in f.readlines():
        record = json.loads(line)

        file_name = record.pop("file_name")
        with open(os.path.join(prefix, split, file_name), mode="rb") as stream:
          image = stream.read()

        (name, _) = os.path.splitext(file_name)
        sink.write({
          "__key__": f"{split}/{name}",
          "jpg": image,
          "json": json.dumps(record).encode("utf-8")
        })

if __name__ == "__main__":
  prefix = "data"

  out = os.path.join("archive", "webdatasets", f"test-00000-of-00001.tar")
  to_webdataset("test", out, prefix=prefix)

  out = os.path.join("archive", "webdatasets", f"train-00000-of-00001.tar")
  to_webdataset("train", out, prefix=prefix)
