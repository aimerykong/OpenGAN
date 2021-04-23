# OpenGAN: Open-Set Recognition via Open Data Generation



Real-world machine learning systems need to analyze novel testing data that differs from the training data. In K-way classification, this is crisply formulated as open-set recognition, core to which is the ability to discriminate open-set data outside the K closed-set classes. Two conceptually elegant ideas for open-set discrimination are: 1) discriminatively learning an open-vs-closed binary discriminator by exploiting  some outlier data as the open-set, and 2) unsupervised learning the closed-set data distribution with a GAN and  using its discriminator as the open-set likelihood function. However, the former generalizes poorly to diverse open test data due to overfitting to the training outliers, which unlikely exhaustively span the open-world. The latter does not work well, presumably due to the instable training of GANs. Motivated by the above, we propose OpenGAN, which addresses the limitation of each approach by combining them with several technical insights. First, we show that a carefully selected GAN-discriminator on some real outlier data already achieves the state-of-the-art. Second, we augment the available set of real open training examples with adversarially synthesized "fake" data. 
Third and most importantly, we build the discriminator over the features computed by the closed-world K-way networks.
Extensive experiments show that OpenGAN significantly outperforms prior open-set methods.


**keywords**: out-of-distribution detection, anomaly detection, open-set recognition, novelty detection, density estimation, generative model, discriminative model, adverserial learning, image classification, semantic segmentation.


If you find our model/method/dataset useful, please cite our work ([arxiv manuscript](https://arxiv.org/abs/2104.02939)):

    @inproceedings{OpenGAN,
      title={OpenGAN: Open-Set Recognition via Open Data Generation},
      author={Kong, Shu and Ramanan, Deva},
      booktitle={preprint: arXiv:2104.02939},
      year={2021}
    }


last update: April, 2021

Shu Kong

aimerykong At g-m-a-i-l dot com
