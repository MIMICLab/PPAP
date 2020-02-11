# PPAPNet
Latent-space-level Image Anonymization
with Adversarial Protector Networks

Along with recent achievements in deep learning empowered by enormous amounts of training data, preserving privacy of an individual related to the gathered data is becoming an essential part of public data collection and publication. Advancements in deep learning threaten traditional image anonymization techniques with model inversion attacks that try to reconstruct the original image from the anonymized image. 

In this paper, we propose a privacy-preserving adversarial protector network (PPAPNet) as an image anonymization tool to convert an image into another synthetic image that is both realistic and immune to model inversion attacks. Our experiments on various datasets show that PPAPNet can effectively convert a sensitive image into a high-quality and attack-immune synthetic image. 

T. Kim and J. Yang, "Latent-space-level Image Anonymization with Adversarial Protector Networks," in IEEE Access.

URL: https://ieeexplore.ieee.org/abstract/document/8744221

![alt text](https://github.com/tgisaturday/PPAP/blob/master/figure1.png)

###Training
```
train_[...].py [dataset_name] [model_name] [previous_iteration(0 if initial training)]
```
