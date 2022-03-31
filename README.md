# ImpliCity: City Modeling from Satellite Images with Deep Implicit Occupancy Fields

![ImpliCity](docs/teaser.jpg?raw=true)

This repository provides the code to train and evaluate ImpliCity, a method that reconstructs digital surface models 
(DSMs) from raw photogrammetric 3D point clouds and ortho-images with the help of an implicit neural 3D scene representation.
It represents the official implementation of the paper:

### [ImpliCity: City Modeling from Satellite Images with Deep Implicit Occupancy Fields](https://arxiv.org/abs/2201.09968)
Corinne Stucker, Bingxin Ke, Yuanwen Yue, Shengyu Huang, Iro Armeni, Konrad Schindler

> Abstract: *High-resolution optical satellite sensors, combined with dense stereo algorithms, have made it possible to reconstruct 
3D city models from space. However, these models are, in practice, rather noisy and tend to miss small geometric features that are 
clearly visible in the images. We argue that one reason for the limited quality may be a too early, heuristic reduction of the 
triangulated 3D point cloud to an explicit height field or surface mesh. To make full use of the point cloud and the underlying images, 
we introduce ImpliCity, a neural representation of the 3D scene as an implicit, continuous occupancy field, driven by learned embeddings 
of the point cloud and a stereo pair of ortho-photos. We show that this representation enables the extraction of high-quality DSMs: 
with image resolution 0.5$\,$m, ImpliCity reaches a median height error of ≈0.7 m and outperforms competing methods, especially w.r.t. 
building reconstruction, featuring intricate roof details, smooth surfaces, and straight, regular outlines.*


**Code coming soon, stay tuned!**
