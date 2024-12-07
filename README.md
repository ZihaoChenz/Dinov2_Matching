## **1. Introduction**

Dinov2 location matching converts an image into a one-dimensional feature using Dinov2. By calculating the similarity between features, the location information of the image is determined within an object.

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
│   │   │   ├── negatives
│   │   │   │   ├── stuttgart_00_000000_000001_leftImg8bit
│   │   │   │   │  ├── stuttgart_00_000000_000020_leftImg8bit.jpg
│   │   │   │   │  ├── stuttgart_00_000000_000021_leftImg8bit.jpg
│   │   │   │   │  ├── ...
...
```


## **3. Self-Dataset Preparation**

### **Download Code**
```bash
git clone https://github.com/ZihaoChenz/Dinov2_Matching.git
```

### **Install Environment**
```bash
pip install -r requirement.txt
```

### **Prepare Dataset**

1. Place your **check dataset** in the directory: `data/(building)/check`
2. Place your **reference dataset** in the directory: `data/(building)/ref`
3. You can rename the building folder as per your requirement.

The file structure should be like this:

```plaintext
├── data
│   ├── building
│   │   ├── building1
│   │   │   ├── check
│   │   │   │   ├── img1.jpg
│   │   │   │   ├── img2.jpg
│   │   │   │   └── img3.jpg
│   │   │   ├── ref
│   │   │   │   ├── img4.jpg
│   │   │   │   ├── img5.jpg
│   │   │   │   └── img6.jpg
│   │   ├── south-building
│   │   │   ├── check
│   │   │   │   ├── img1.jpg
│   │   │   │   ├── img2.jpg
│   │   │   │   └── img3.jpg
│   │   │   ├── ref
│   │   │   │   ├── img4.jpg
│   │   │   │   ├── img5.jpg
│   │   │   │   └── img6.jpg
│   ├── surrounding
│   │   ├── Cyberport
│   │   │   ├── check
│   │   │   │   ├── img1.jpg
│   │   │   │   ├── img2.jpg
│   │   │   │   └── img3.jpg
│   │   │   ├── ref
│   │   │   │   ├── img4.jpg
│   │   │   │   ├── img5.jpg
│   │   │   │   └── img6.jpg
│   │   ├── HKU
│   │   │   ├── check
│   │   │   │   ├── img1.jpg
│   │   │   │   ├── img2.jpg
│   │   │   │   └── img3.jpg
│   │   │   ├── ref
│   │   │   │   ├── img4.jpg
│   │   │   │   ├── img5.jpg
│   │   │   │   └── img6.jpg
...
```

---

## **4. Inference Data**
If need to inference ZuBuD dataset:
Run a script to inference ZuBuD
```bash
python scripts/inference_ZuBuD.py
```
The output results after inference are saved to output/ZuBuD by default.

Else:
Run the following command to start the inference:
```bash
python inference.py --CheckFolder data/... --OutputFolder xxx/xxx
```
- **CheckFolder**: Path to the folder containing the check images
- **OutputFolder**: Path to save the output

Example:
```bash
python inference.py --CheckFolder data/building --OutputFolder output/building
```

After inference, the file structure will look like this:

```plaintext
├── output
│   ├── building
│   │   ├── building1
│   │   │   ├── check
│   │   │   │   ├── img1.txt
│   │   │   │   ├── img2.txt
│   │   │   │   └── img3.txt
│   │   │   ├── ref
│   │   │   │   ├── img4.txt
│   │   │   │   ├── img5.txt
│   │   │   │   └── img6.txt
│   │   ├── south-building
│   │   │   ├── check
│   │   │   │   ├── img1.txt
│   │   │   │   ├── img2.txt
│   │   │   │   └── img3.txt
│   │   │   ├── ref
│   │   │   │   ├── img4.txt
│   │   │   │   ├── img5.txt
│   │   │   │   └── img6.txt
│   ├── surrounding
│   │   ├── Cyberport
│   │   │   ├── check
│   │   │   │   ├── img1.txt
│   │   │   │   ├── img2.txt
│   │   │   │   └── img3.txt
│   │   │   ├── ref
│   │   │   │   ├── img4.txt
│   │   │   │   ├── img5.txt
│   │   │   │   └── img6.txt
│   │   ├── HKU
│   │   │   ├── check
│   │   │   │   ├── img1.txt
│   │   │   │   ├── img2.txt
│   │   │   │   └── img3.txt
│   │   │   ├── ref
│   │   │   │   ├── img4.txt
│   │   │   │   ├── img5.txt
│   │   │   │   └── img6.txt
```

---

## **5. Visualization (Normal Method)**

To visualize the results, use the following command:
```bash
python visualize.py --ImageType xxx --ResultFolder xxx/xxx --DataFolder xxx/xxx
```
- **ImageType**: Image format (e.g., `jpg`, `png`)
- **ResultFolder**: The folder where the inference results are saved
- **DataFolder**: The folder containing the original dataset images

You can convert the image format using `utils.convert_image_format.py` if needed.

Example:
```bash
python visualize.py --ImageType JPG --ResultFolder output/building/building1 --DataFolder data/building/building1
```

---

## **6. Centroid Method for Matching**

### **Create Embeddings**

To create embeddings, run:
```bash
python create_embedding.py --Embedding_target_folder xxx/xxx --Embedding_save_folder xxx/xxx
```
- **Embedding_target_folder**: Folder containing data to generate embeddings (e.g., `output/building`)
- **Embedding_save_folder**: Folder to save the generated embeddings (e.g., `embedding/building`)

### **Visualize (Centroid Method)**

```bash
python visualize.py --ImageType xxx --GalleryData xxx/xxx --CheckTxtFolder xxx/xxx --OutputBaseFolder xxx/xxx
```
- **GalleryData**: Path to the embedding data folder
- **CheckTxtFolder**: Path to the folder containing check `.txt` files
- **OutputBaseFolder**: Path to the base folder of the check images
- Use `--Normalize` to normalize features if necessary

Example:
```bash
python visualize.py --ImageType jpg --GalleryData embedding/surrounding --CheckTxtFolder output/surrounding/Cyberport/check --OutputBaseFolder output/surrounding
```
