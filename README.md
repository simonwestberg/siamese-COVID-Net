# Siamese-COVID-Net

In this project we implement a siamese convolutional neural network based on the COVID-Net architecture introduced by Wang et al. (https://www.nature.com/articles/s41598-020-76550-z) for classification of x-ray images of lungs infected with COVID-19. The project was done as part of the course DD2424 at KTH. The report detailing our method and findings can be found [here](report.pdf).

## Architecture 

The architecture of the COVID-Net is illustrated in the image below. 

![COVID-Net-architecure](/figures/COVID-Net-architecture.png)

Each PEPX module consists of a 

* First-stage Projection: a 1×1 convolution to project data into a tensor with a lower channel dimension

* Expansion: a 1×1 convolution used to expand the number of channels

* DW Convolution: a 3×3 depth-wise convolution

* Second-stage Projection: a 1×1 convolution to project data into a tensor with a lower channel dimension

* Extension: a 1×1 convolution to extend the number of channels in the output

![pepx](/figures/pepx.png)

The COVID-Net is duplicated and used as a component in our siamese network, illustrated below

![siamese-net](/figures/siamese-arch.png)
