<div align="center">
<h3 align="center">ICDAR's Scanned Receipts OCR and Information Extraction (SROIE)</h3>
  <p align="center">
    Hugging Face Dataset based on the 15th International Conference on Document
    Analysis and Recognition (ICDAR2019) Robust Reading Challenge on Scanned
    Receipts OCR and Information Extraction (SROIE).
    <br />
    <br />
    <a href="https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE">
      <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-xl-dark.svg" alt="Dataset on HF" />
    </a>
  </p>
</div>


## About The Project

This project contains the processing code and dataset for an extension of the original ICDAR2019 SROIE dataset, including the following changes/additions:

* Integrated with Hugging Face's `datasets` format
* Addition of 14 extracted texts missing from the Task 3 test dataset covering
  key information extraction
* Corrections and improvements to the original dataset (see [Modifications to the Dataset](#modifications-to-the-dataset))

The dataset card and distribution (Parquet-format) can be found on Hugging Face:
https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE

<iframe
  src="https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE/embed/viewer/default/train"
  frameborder="0"  width="100%"  height="560px">
</iframe>

The ICDAR2019 SROIE dataset was originally published by Huang et al. for the
[15th International Conference on Document Analysis and Recognition (ICDAR2019)
Robust Reading Challenge on Scanned Receipts OCR and Information Extraction
(SROIE)](https://rrc.cvc.uab.es/?ch=13).


## Getting Started

### Prerequisites

Install the required Python packages:

```sh
python3 -m pip install datasets torch torchvision
```

### Using this Dataset

You can load and visualize the dataset either directly from Hugging Face
(recommended) or offline (see [Loading Dataset
Offline](#loading-dataset-offline) below).

```python
import torch
from datasets import load_dataset
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

# The recommended method of loading this dataset is from Hugging Face
dataset = load_dataset("jsdnrs/SROIE2019", split="all")

# However, the dataset can be loaded offline using the "imagefolder" loader
# once the images are downloaded (see below)
#dataset = load_dataset("imagefolder", data_dir="data", split="all")

# Print an example of the extracted text
print(dataset[0]["entities"])

# Show an example receipt image with the bounding boxes overlaid
img = to_pil_image(
  draw_bounding_boxes(
    pil_to_tensor(dataset[0]["image"]),
    torch.tensor(dataset[0]["bboxes"]),
    colors="red"))
img.show()
```

Example output:

```json
{
  "company": "BOOK TA .K (TAMAN DAYA) SDN BHD",
  "date": "25/12/2018",
  "address": "NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA, 81100 JOHOR BAHRU, JOHOR.",
  "total": "9.00"
}
```

<div style="text-align: center;">
  <img src="assets/ex.jpg" alt="Annotated receipt image" style="width: 250px; margin: auto; display: block;">
</div>

### Loading Dataset Offline

If you want to work with the dataset locally:

1. Download the image files from Google Drive (available in TAR, ZIP, and
   webdataset formats): https://drive.google.com/drive/folders/1j3VFuWDSdFSmVZwSwm1wXgKxy4VKiicc

2. Unpack them into the folder structure:

```
data/
  train/
  test/
```

3. Use the `imagefolder` loader from `datasets`:

```python
from datasets import load_dataset
dataset = load_dataset("imagefolder", data_dir="data", split="all")
```


## Modifications to the Dataset

The following modifications were made to improve the original ICDAR2019 SROIE
dataset:

* All meta information has been stripped from the original images with
  [`exiftool`](https://exiftool.org/) using: `exiftool -all:all= ...`

* The bounding box (bbox) coordinates originally given for Tasks 1 & 2 were
  given as "rectangles with four vertices, which are in clockwise order starting
  from the top" in the format: [x1, y1, x2, y2, x3, y3, x4, y4]. These bbox
  coordinates have been reduced to a [x1, y1, x3, y3] format, representing two
  corners with (x1, y1) being top left and (x3, y3) being bottom right.
  * This modification more easily integrates with `torchvision`.

* A set of invalid UTF-8 characters were removed from the original Task 2
  transcript of the X51006619503.jpg image. There did not appear to be any
  associated text in the image corresponding to these characters.

* A additional 14 extracted texts for the Task 3 test data set were added by the
  author of this dataset. These extracted texts are included in this repository
  under `additional_data/test/task3/`.


## License

Distributed under the [Creative Commons Attribution 4.0 International License
(CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/). See `LICENSE` and
`NOTICE` for more information.


## Acknowledgments

TODO: Add acknowledgements to other existing datasets.
