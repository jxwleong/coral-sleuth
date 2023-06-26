﻿# coral-sleuth ![main workflow](https://github.com/jxwleong/coral-sleuth/actions/workflows/main.yml/badge.svg)
## <a name="toc"></a> Table of Contents
- [Dataset](#dataset)



<br/><br/>
<!-- omit in toc -->
## <a name="dataset"></a> Dataset [<sub><sup>Back to Table of Contents</sup></sub>](#toc)

The data used in this project is a combination of two datasets from CoralNet and MCR LTER. It is designed to facilitate the task of coral reef identification, and it's publicly available on Kaggle [here](https://www.kaggle.com/datasets/jxwleong/coral-reef-dataset).

<br/>

### Source 1: CoralNet
CoralNet (source: [CoralNet Source](https://coralnet.ucsd.edu/source/2091/)) is a place where scientists and researchers share pictures of coral reefs. These pictures come from all over the world, and the one we are using for our project is from around Okinawa, which is an island in Japan.

The pictures in this dataset show different types of corals, underwater creatures, and even algae. Each picture also has labels that tell you what's in the picture. This is really helpful if you're trying to teach a computer to recognize different types of corals and underwater life.

Another great thing about CoralNet is that it lets people work together to label the pictures. This means more people checking and making sure the labels are correct, which makes the data better for everyone.

<br/>

### Source 2: MCR LTER 
MCR LTER (source: ) The Moorea Coral Reef LTER is a subset of the MCR LTER dedicated to fostering extensive research in the field of coral reef ecosystems. The dataset includes 2055 images captured from three distinct habitats over the years 2008, 2009, and 2010.

The images feature annotations for nine primary labels, four of which are non-coral: Crustose Coralline Algae (CCA), Turf Algae, Macroalgae, and Sand, and five are coral genera: Acropora, Pavona, Montipora, Pocillopora, and Porites. These labels cover a whopping 96% of all the annotations, resulting in nearly 400,000 points.

This data subset has been employed in numerous computer vision research projects and publications, including 'Automated Annotation of Coral Reef Survey Images', presented at the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) in 2012.

The research and collection of these data have been made possible through the support of the U.S. National Science Foundation, along with a significant contribution from the Gordon and Betty Moore Foundation. It also received approval from the French Polynesian Government, which highlights its value and relevance in the field of coral reef research.

<br/>

### Data Composition
The dataset contains **[4385]** images, where each image is associated with a label indicating the type of coral. There are a total of **[8]** distinct coral types/classes in the dataset. The data has been preprocessed and split into training and validation sets. Each image has a corresponding position coordinate.

<br/>

### Usage
This dataset is used in our project to train and evaluate the performance of our coral reef classification model. By leveraging these rich, annotated datasets, the model can learn to identify different types of coral reefs from images and associated position data.

Please refer to the Kaggle page linked above for download instructions and more detailed information about the data.
