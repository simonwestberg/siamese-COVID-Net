# Siamese COVID-Net

In this project we implement a siamese convolutional neural network based on the COVID-Net architecture introduced by Wang et al. (https://www.nature.com/articles/s41598-020-76550-z) for classification of X-ray images of lungs that are either healthy, infected with COVID-19, or infected with pneumonia. We compare the performance of our siamese network to a regular CNN based on COVID-Net and analyze the network predictions using [GradCAM](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html). The networks were implemented in TensorFlow using Keras and the project was done as part of the course DD2424 Deep Learning at KTH. The report detailing our method and findings can be seen [here](report.pdf).

## COVID-Net Architecture 

The architecture of COVID-Net is illustrated in the image below. It is primarily based on repeated use of the computationally efficient PEPX module together with skip connections. ReLU activation is applied to all layers and downsampling is done by 2×2 max pooling.

![COVID-Net-architecure](/figures/COVID-Net-architecture.png)
*The architecture of our implementation of COVID-Net. Used as a component in the siamese network.*

Each PEPX (projection-expansion-projection-extension) module consists of a 

* First-stage Projection: a 1×1 convolution projecting the input to a lower channel dimension

* Expansion: a 1×1 convolution expanding the number of channels

* DW Convolution: a 3×3 depth-wise convolution

* Second-stage Projection: a 1×1 convolution reducing the channel dimension once again

* Extension: a 1×1 convolution extending the number of channels once again

![pepx](/figures/pepx.png)
*The PEPX module used in COVID-Net*

## Siamese network

A siamese network structure is perhaps most easily explained as two ordinary CNNs working side-by-side,
where the “siamese” part comes from the fact that the two networks are identical and share the same weights. 
As two images are passed through the networks, their corresponding feature vectors are calculated and a distance metric is used 
to determine the similarity of the two images. An illustration of a siamese network can be seen below.

![siamese-net](/figures/siamese-arch.png)
*Typical structure of a siamese network. The function d denotes a distance measure used to determine the similarity between the two images.*

