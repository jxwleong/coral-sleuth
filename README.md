# coral-sleuth ![main workflow](https://github.com/jxwleong/coral-sleuth/actions/workflows/main.yml/badge.svg) 

## <a name="toc"></a> Table of Contents
- [Overview](#overview)
- [Model](#model)
  - [EfficientNetV2](#efficient-net-v2)
  - [MobileNetV3](#mobilenet-v3)
  - [ConvNet](#convnet)
  - [Model Selection and Training](#model-selection-and-training)
- [Dataset](#dataset)
- [Environment and Setup](#environment-and-setup)
  - [Setup NVIDIA GPU][#setup-nvidia-gpu]
- [Reference](#reference)
  


<br/><br/>
<!-- omit in toc -->
## <a name="overview"></a> Overview [<sub><sup>Back to Table of Contents</sup></sub>](#toc)
"Coral Sleuth" is a deep learning project focused on advancing the precise and efficient identification of coral species from underwater images. The core ambition of the project is to foster a technological contribution that can augment the speed and accuracy of coral identification, thereby enabling better management and conservation of precious coral reef ecosystems.

At the heart of the project lies a novel deep learning model, engineered specifically for the task of coral identification. The model is designed with a keen focus on balancing computational efficiency - allowing for real-time analysis and scalability for large-scale deployments - with high predictive accuracy, crucial for reliable identification of diverse coral species.

The project comprises the following key components:

1. Data Collection: Gathering a comprehensive and robust dataset of high-resolution underwater images of various coral species.

2. Model Benchmarking: Evaluating and comparing the performance of a variety of state-of-the-art deep learning models in terms of computational efficiency and predictive accuracy.

3. Model Development: Creating a novel, optimized deep learning model for coral identification, based on insights gained from benchmarking and tailored for the project's unique requirements.

4. Model Validation: Rigorously testing the model's performance on unseen datasets, to ensure its robustness, generalizability, and reliability in diverse coral ecosystems.

Through these elements, "Coral Sleuth" aims to provide a valuable tool that can assist in the study, preservation, and management of coral species, supporting global efforts to protect and conserve our invaluable coral reef ecosystems.

<br/><br/>
<!-- omit in toc -->
## <a name="model"></a> Model [<sub><sup>Back to Table of Contents</sup></sub>](#toc)
In this project, I will initially focus on training three state-of-the-art models, EfficientNet V2, MobileNet V3, and ConvNeXt, on a subset of the complete dataset. These models have been chosen due to their established capabilities and strong performance in a variety of image classification tasks.

By starting with a smaller dataset, my goal is to quickly evaluate and compare the performance of these three models. This approach allows me to save computational resources while rapidly identifying the most promising model for the specific task at hand.

Once the best performing model is identified based on its accuracy, precision, recall, and other key metrics, the next step will be to fine-tune this model using the complete dataset. Fine-tuning is expected to further enhance the model's performance, improving its capability to classify coral reef images accurately and efficiently.

In future steps, I plan to continually assess the model's performance, making necessary adjustments to optimize its predictions. This iterative process is crucial for ensuring the model's ongoing learning and adaptation, thereby improving its performance and contributing to the advancement of coral reef classification and conservation efforts.

| Model            | Size (MB) | Top-1 Accuracy | Top-5 Accuracy | Parameters | Depth | Time (ms) per inference step (CPU) | Time (ms) per inference step (GPU) |
|------------------|-----------|----------------|----------------|------------|-------|------------------------------------|------------------------------------|
| EfficientNet V2B0| 29        | 78.7%          | 94.3%          | 7.2M       | -     | -                                  | -                                  |
| EfficientNet V2B1| 34        | 79.8%          | 95.0%          | 8.2M       | -     | -                                  | -                                  |
| EfficientNet V2B2| 42        | 80.5%          | 95.1%          | 10.2M      | -     | -                                  | -                                  |
| EfficientNet V2B3| 59        | 82.0%          | 95.8%          | 14.5M      | -     | -                                  | -                                  |
| EfficientNet V2S | 88        | 83.9%          | 96.7%          | 21.6M      | -     | -                                  | -                                  |
| EfficientNet V2M | 220       | 85.3%          | 97.4%          | 54.4M      | -     | -                                  | -                                  |
| EfficientNet V2L | 479       | 85.7%          | 97.5%          | 119.0M     | -     | -                                  | -                                  |
| MobileNet V3     | -         | -              | -              | -          | -     | -                                  | -                                  |
| ConvNeXtTiny     | 109.42    | 81.3%          | -              | 28.6M      | -     | -                                  | -                                  |
| ConvNeXtSmall    | 192.29    | 82.3%          | -              | 50.2M      | -     | -                                  | -                                  |
| ConvNeXtBase     | 338.58    | 85.3%          | -              | 88.5M      | -     | -                                  | -                                  |
| ConvNeXtLarge    | 755.07    | 86.3%          | -              | 197.7M     | -     | -                                  | -                                  |
| ConvNeXtXLarge   | 1310      | 86.7%          | -              | 350.1M     | -     | -                                  | -                                  |

Reference: https://keras.io/api/applications/ 

<br/><br/>

### <a name="efficient-net-v2"></a> EfficientNet V2 [<sub><sup>Back to Table of Contents</sup></sub>](#toc)
### <a name="mobilenet-v3"></a> MobileNet V3 [<sub><sup>Back to Table of Contents</sup></sub>](#toc)
### <a name="convnet"></a> ConvNeXt [<sub><sup>Back to Table of Contents</sup></sub>](#toc)

<br/><br/>
<!-- omit in toc -->
### <a name="model-selection-and-training"></a> 
Model Selection and Training [<sub><sup>Back to Table of Contents</sup></sub>](#toc)  


<br/><br/>
<!-- omit in toc -->
## <a name="dataset"></a> Dataset [<sub><sup>Back to Table of Contents</sup></sub>](#toc)

The data used in this project is a combination of two datasets from CoralNet and MCR LTER. It is designed to facilitate the task of coral reef identification, and it's publicly available on Kaggle [here](https://www.kaggle.com/datasets/jxwleong/coral-reef-dataset).

<br/>

### Source 1: [CoralNet](https://coralnet.ucsd.edu/source/2091/) 
CoralNet is a place where scientists and researchers share pictures of coral reefs. These pictures come from all over the world, and the one we are using for our project is from around Okinawa, which is an island in Japan.

The pictures in this dataset show different types of corals, underwater creatures, and even algae. Each picture also has labels that tell you what's in the picture. This is really helpful if you're trying to teach a computer to recognize different types of corals and underwater life.

Another great thing about CoralNet is that it lets people work together to label the pictures. This means more people checking and making sure the labels are correct, which makes the data better for everyone.

Label List: https://coralnet.ucsd.edu/label/list/

<br/>

### Source 2: [MCR LTER](http://mcr.lternet.edu/cgi-bin/showDataset.cgi?docid=knb-lter-mcr.5006)
The Moorea Coral Reef LTER is a subset of the MCR LTER dedicated to fostering extensive research in the field of coral reef ecosystems. The dataset includes 2055 images captured from three distinct habitats over the years 2008, 2009, and 2010.

The images feature annotations for nine primary labels, four of which are non-coral: Crustose Coralline Algae (CCA), Turf Algae, Macroalgae, and Sand, and five are coral genera: Acropora, Pavona, Montipora, Pocillopora, and Porites. These labels cover a whopping 96% of all the annotations, resulting in nearly 400,000 points.

This data subset has been employed in numerous computer vision research projects and publications, including 'Automated Annotation of Coral Reef Survey Images', presented at the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) in 2012.

The research and collection of these data have been made possible through the support of the U.S. National Science Foundation, along with a significant contribution from the Gordon and Betty Moore Foundation. It also received approval from the French Polynesian Government, which highlights its value and relevance in the field of coral reef research.

<br/>

### Data Composition
The dataset contains **[4385]** images, where each image is associated with a label indicating the type of coral. There are a total of **[8]** distinct coral types/classes in the dataset. The data has been preprocessed and split into training and validation sets. Each image has a corresponding position coordinate.

Full combined annotation label distribution of [combined_annotations_remapped.csv](data/annotations/combined_annotations_remapped.csv)
<details>   
<summary>Click to expand!</summary>

```
Label distribution:
crustose_coralline_algae    226017
turf                         43769
sand                         38880
porites                      35236
macroalgae                   23832
off                          13605
pocillopora                  11319
montipora                     8755
pavona                        5806
acropora                      3458
hard_substrate                2086
millepora                     1459
broken_coral_rubble           1025
montastraea                    645
leptastrea                     528
soft                           280
bad                            259
goniastrea                     198
dark                           191
fungia                         160
algae                          142
astreopora                     129
gardineroseris                 123
herpolitha                      81
dead_coral                      52
favia                           47
lobophyllia                     47
soft_coral                      38
platygyra                       26
rock                            24
echinopora                      24
cyphastrea                      18
acanthastrea                    16
green_fleshy_algae               8
psammocora                       7
stylophora                       7
favites                          6
leptoseris                       4
sandolitha                       2
tuba                             1
```
</details>   

<br/>

### Usage
This dataset is used in our project to train and evaluate the performance of our coral reef classification model. By leveraging these rich, annotated datasets, the model can learn to identify different types of coral reefs from images and associated position data.

Please refer to the Kaggle page linked above for download instructions and more detailed information about the data.

<br/><br/>
### <a name="environment-and-setup"></a> Environment and Setup] [<sub><sup>Back to Table of Contents</sup></sub>](#toc)
#### <a name="setup-nvidia-gpu"></a> Setup NVIDIA GPU [<sub><sup>Back to Table of Contents</sup></sub>](#toc)
---
 ***NOTE***: If you are training the model in Windows. There are specific version of the libraries or
 toolkit you have to used as specified in [[4]](https://www.tensorflow.org/install/pip#windows-native_1).

 - Tensorflow: "Anything above 2.10 is not supported on the GPU on Windows Native". Use command `pip install "tensorflow<2.11"` to install.
 - CUDA Toolkit: [v11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive)
 - cuDNN: [v8.1.0](https://developer.download.nvidia.com/compute/redist/cudnn/v8.1.0/)

---

Setting up your environment to run deep learning models requires some specific steps, especially when using a GPU for computation. In this section, we will cover the steps required to set up an NVIDIA GPU on a Windows machine.

1. **Check your GPU**: Before setting up the GPU, you must confirm whether your system has an NVIDIA GPU with CUDA support. You can do this by checking this website https://developer.nvidia.com/cuda-gpus.
2. **Install NVIDIA GPU driver**: Visit the NVIDIA Driver Downloads page (https://www.nvidia.com/download/index.aspx), select your GPU model from the list, and download the driver. Run the downloaded file and follow the prompts to install the driver. Reboot your computer once the installation is complete.
3. **Install CUDA Toolkit**: CUDA is a parallel computing platform and programming model developed by NVIDIA for general computing on GPUs. You can download the CUDA Toolkit from the NVIDIA website (https://developer.nvidia.com/cuda-toolkit). Be sure to select the version that is compatible with your system and the deep learning framework you plan to use (TensorFlow, PyTorch, etc.).
4. **Install cuDNN**: The NVIDIA CUDA Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks, which is also required for running most of the deep learning frameworks. After downloading the appropriate version from the NVIDIA cuDNN page (https://developer.download.nvidia.com/compute/redist/cudnn/), you can install it by copying the extracted files to the CUDA Toolkit directory.
5. **Configure the Environment Variables**: After successfully installing CUDA and cuDNN, add their bin directories to the PATH environment variable. You can do this by navigating to 'Environment Variables' in your system settings, and then appending the paths of CUDA and cuDNN to the PATH variable.
6. **Verify the installation**:
   - Verify the CUDA toolkit and cuDNN installation: 
     - Open the command prompt and type nvcc -V. This should return the CUDA compiler version if the installation was successful. You could also verify the installation by running a simple CUDA program or a deep learning model using your chosen framework. 
   - Verify Tensorflow package installation:
     - Run the following command in CMD:
       - `python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())`. If it returns `True`, then the installation is correct. Otherwise, try to reinstall the tensorflow package via pip.
       - `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))`. This command will return the information about the GPU of the system. If it return empty list (`[]`), please try to reinstall tensorflow package.


<br/><br/>

## <a name="reference"></a> Reference [<sub><sup>Back to Table of Contents</sup></sub>](#toc)
1. [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298v3)
2. [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
3. [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
4. [Install TensorFlow with pip: Windows Native](https://www.tensorflow.org/install/pip#windows-native_1)