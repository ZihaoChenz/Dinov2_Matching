## **1. Install Environment**

```bash
pip install -r requirement.txt
```

---

## **2. Prepare Training Dataset**

### **Download Online Dataset**
1. Download CityScapes dataset by https://www.cityscapes-dataset.com/file-handling/?packageID=12
2. Select the whole data or a subset of data as the dataset for the project.

### **Prepare Positive Data**
1. Using projects that can transform photos into different styles and weather conditions to build a positive sample set.
2. You can use CCPL to generate (https://github.com/JarrentWu1031/CCPL)

### **Build Complete Dataset**
1. Place the images you want to query into `datasets/(dataset name)/train/queries and datasets/(dataset name)/val/queries`.
2. Place the positive sample set for each corresponding frame image into `datasets/(dataset name)/train/positives` and `datasets/(dataset name)/val/positives`.
3. Put the corresponding negative samples for the query images into `datasets/(dataset name)/train/negatives` and `datasets/(dataset name)/val/negatives`, such as images that are more than 10 frames apart from the query image in the dataset.

The file structure should be like this:

```plaintext
├── datasets
│   ├── cityscapes
│   │   ├── train
│   │   │   ├── queries
│   │   │   │   ├── stuttgart_00_000000_000001_leftImg8bit.jpg
│   │   │   │   ├── stuttgart_00_000000_000009_leftImg8bit.jpg
│   │   │   ├── positives
│   │   │   │   ├── stuttgart_00_000000_000001_leftImg8bit
│   │   │   │   │  ├── stuttgart_00_000000_000001_leftImg8bit_stylized_foggy149.jpg
│   │   │   │   │  ├── stuttgart_00_000000_000001_leftImg8bit_stylized_rain275.jpg
│   │   │   │   ├── stuttgart_00_000000_000009_leftImg8bit
│   │   │   │   │  ├── stuttgart_00_000000_000009_leftImg8bit_stylized_foggy149.jpg
│   │   │   │   │  ├── stuttgart_00_000000_000009_leftImg8bit_stylized_rain275.jpg
│   │   │   │   ├── ...
│   │   │   ├── negatives
│   │   │   │   ├── stuttgart_00_000000_000001_leftImg8bit
│   │   │   │   │  ├── stuttgart_00_000000_000020_leftImg8bit.jpg
│   │   │   │   │  ├── stuttgart_00_000000_000021_leftImg8bit.jpg
│   │   │   │   │  ├── ...
│   │   │   │   ├── stuttgart_00_000000_000009_leftImg8bit
│   │   │   │   │  ├── stuttgart_00_000000_000021_leftImg8bit.jpg
│   │   │   │   │  ├── stuttgart_00_000000_000022_leftImg8bit.jpg
│   │   │   │   │  ├── ...
│   │   │   │   ├── ...
...
```

## **3. Train Model**
Run train.py
```bash
train.py --DataFolder datasets/xx --CheckPoints checkpoints/xx --batch_size x --epoch x
```
For Example:
```bash
train.py --DataFolder datasets/cityscapes --CheckPoints checkpoints/cityscapes --batch_size 8 --epoch 200
```


## **4. Inference Preparation**

### **Prepare Data**

1. Place your **check data** in the directory: `data/xx/check`
2. Place your **reference data** in the directory: `data/xx/ref`
3. You can rename the building folder as per your requirement.

For example:
The file structure should be like this:

```plaintext
├── data
│   ├── cityscapes
│   │   ├── sunrise
│   │   │   ├── check
│   │   │   │   ├── stuttgart_00_000000_000001_leftImg8bit.jpg
│   │   │   │   ├── stuttgart_00_000000_000009_leftImg8bit.jpg
│   │   │   │   └── ...
│   │   │   ├── ref
│   │   │   │   ├── stuttgart_00_000000_000001_leftImg8bit.jpg
│   │   │   │   ├── stuttgart_00_000000_000001_leftImg8bit_stylized_sunrise21.jpg
│   │   │   │   ├── stuttgart_00_000000_000002_leftImg8bit.jpg
│   │   │   │   ├── stuttgart_00_000000_000002_leftImg8bit_stylized_sunrise21.jpg
│   │   │   │   └── ...
```

---

## **5. Inference Data**
Run the following command to start the inference:
```bash
python inference.py --CheckFolder data/... --OutputFolder ouput/xxx
```
- **CheckFolder**: Path to the folder containing the check images
- **OutputFolder**: Path to save the output
- **model**: Path of your own model, default is dinov2 pretrained model

Example:
```bash
python inference.py --CheckFolder data/cityscapes/sunrise --OutputFolder output/cityscapes/sunrise
```

After inference, the file structure will look like this:

```plaintext
├── output
│   ├── cityscapes
│   │   ├── sunrise
│   │   │   ├── check
│   │   │   │   ├── stuttgart_00_000000_000001_leftImg8bit.txt
│   │   │   │   ├── stuttgart_00_000000_000009_leftImg8bit.txt
│   │   │   │   └── ...
│   │   │   ├── ref
│   │   │   │   ├── stuttgart_00_000000_000001_leftImg8bit.txt
│   │   │   │   ├── stuttgart_00_000000_000001_leftImg8bit_stylized_sunrise21.txt
│   │   │   │   ├── stuttgart_00_000000_000002_leftImg8bit.txt
│   │   │   │   ├── stuttgart_00_000000_000002_leftImg8bit_stylized_sunrise21.txt
│   │   │   │   └── ...
```

---

## **6. Visualization**

To visualize the results, use the following command:
```bash
python visualize.py --ImageType xxx --ResultFolder xxx/xxx --DataFolder xxx/xxx
```
- **ImageType**: Image format (e.g., `jpg`, `png`)
- **ResultFolder**: The folder where the inference results are saved
- **DataFolder**: The folder containing the original dataset images
- **json**: Record the similarity between each query image and all other reference (ref) images

You can convert the image format using `utils.convert_image_format.py` if needed.

Example:
```bash
python visualize.py --ImageType JPG --ResultFolder output/cityscapes/sunrise --DataFolder data/cityscapes/sunrise
```

---

## **7. Evaluation**
To evaluate the results, use the following command:
```bash
python evaluation.py --json_path xxx
```
- **json_path**: The path of json file
