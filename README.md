# Dinov2_Matching
**Dinov2 Location Matching Report**

1. **Introduction**

Dinov2 location matching converts the image into a one-dimensional feature through Dinov2, and determines the location information of the image in the object by calculating the similarity.

2. **Preparation**

- Downloads code:
    - git clone https://github.com/ZihaoChenz/Dinov2_Matching.git
- Prepare dataset:
    - Put your own check dataset in data/(building)/check.
    - Put your own reference dataset in data/(building)/ref.
    - You can change the building to any name you want.
    - The file structure will be:

For example:

├──data

│ ├──building

││ ├──building1

││ │ ├──check

││ │ │ ├──img1.jpg

││ │ │ ├──img2.jpg

││ │ │ └──img3.jpg

││ │ ├──ref

││ │ │ ├──img4.jpg

││ │ │ ├──img5.jpg

││ │ │ └──img6.jpg

││ ├──south-building

││ │ ├──check

││ │ │ ├──img1.jpg

││ │ │ ├──img2.jpg

││ │ │ └──img3.jpg

││ │ ├──ref

││ │ │ ├──img4.jpg

││ │ │ ├──img5.jpg

││ │ │ └──img6.jpg

│ ├──surrounding

││ ├──Cyberport

││ │ ├──check

││ │ │ ├──img1.jpg

││ │ │ ├──img2.jpg

││ │ │ └──img3.jpg

││ │ ├──ref

││ │ │ ├──img4.jpg

││ │ │ ├──img5.jpg

││ │ │ └──img6.jpg

- Inference data:
      ```python inference.py --CheckFolder data/... --OutputFolder xxx/xxx```
    - Reference folder default is in output
    - For example
      ```python inference.py --CheckFolder data/building --OutputFolder output/building```
    - After inference, the file structure will be:

For example:

├──data

│ ├──building

││ ├──building1

││ │ ├──check

││ │ │ ├──img1.jpg

││ │ │ ├──img2.jpg

││ │ │ └──img3.jpg

││ │ ├──ref

││ │ │ ├──img4.jpg

││ │ │ ├──img5.jpg

││ │ │ └──img6.jpg

├──output

│ ├──building

││ ├──building1

││ │ ├──check

││ │ │ ├──img1.txt

││ │ │ ├──img2.txt

││ │ │ └──img3.txt

││ │ ├──ref

││ │ │ ├──img4.txt

││ │ │ ├──img5.txt

││ │ │ └──img6.txt

- Visualize (for normal method):
      ```python visualize.py --ImageType xxx --ResultFolder xxx/xxx --DataFolder xxx/xxx```
    - xxx is your dataset images type, such as jpg, png, you can through utils.convert_image_format.py to convert your image format
    - You need to select your result folder which is output by inference.
    - Select the data folder path corresponding to the result folder.
    - For example
      ```python visualize.py --ImageType JPG --ResultFolder output/building/building1 --DataFolder data/building/building1```

1. **Use centorid method to match image**
- Create Embedding:
      ```python create_embedding.py --Embedding_target_folder xxx/xxx --Embedding_save_folder xxx/xxx```
    - Embedding_target_folder is the folder that you want to generate embedding, for example: output/building
    - Embedding_save_folder is the path that save output embedding, for example: embedding/building
- Visualize (for centroid method):
      ```python --ImageType xxx --GalleryData xxx/xxx --CheckTxtFolder xxx/xxx --OutputBaseFolder xxx/xxx```
    - GalleryData is the embedding data folder
    - CheckTxtFolder is the txt folder path that you want to check
    - OutputBaseFolder is the folder that the check folder belong to
    - You can choose to normalize the feature by using --Normalize
    - For example:
      ```python --ImageType jpg --GalleryData embedding/surrounding --CheckTxtFolder output/surrounding/Cyberport/check --OutputBaseFolder output/surrounding```
