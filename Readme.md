# Black and White image Colorization with GAN and U-Net

# Papers and Strategy used 
```bash
https://arxiv.org/abs/1603.08511
```
```bash
https://arxiv.org/abs/1611.07004 
```
Also I used Few code snippets from some online available resources

# Note
1. Every epoch takes about 7 to 48 minutes on Colab. After about 20 epochs you should see some reasonable results hence I used Tesla T4 GPU hence it is suggested to use some good GPU as I stopped after 6-7 epochs as it was about to take 12 hours+ to train the model.
2. The paper utilizes the entire ImageNet dataset, which consists of 1.3 million images, but for this project, we are only using 8,000 images from the COCO dataset. This means that our training set size is just 0.6% of the size used in the paper. However, it's important to note that you can use almost any dataset for this task, as long as it contains a variety of scenes and locations that the model can learn to colorize. For example, you could use ImageNet, but you would only need 8,000 of its images for this project.

# Model Architecture

1. When loading an image, we get a rank-3 array with the last axis containing the color data in RGB color space. However, when training a model for colorization, it's more convenient to use `L*a*b` color space instead of RGB. The reason is that in L*a*b , we have three channels `(L, *a, and *b)` where L encodes the lightness and the other two channels encode the green-red and yellow-blue components, respectively. By giving the L channel to the model and hoping it predicts the other two channels, we can concatenate all the channels and get a colorful image. On the other hand, using RGB would require converting the image to grayscale, feeding it to the model, and hoping it predicts three numbers for each pixel, which is a more difficult and unstable task due to the many more possible combinations of three numbers compared to two numbers.

2. `pix2pix`  proposed a general solution to many image-to-image tasks in deep learning which one of those was colorization. In this approach two losses are used: L1 loss, which makes it a regression task, and an adversarial (GAN) loss, which helps to solve the problem in an unsupervised manner.

3. The code implements a U-Net as the generator of a GAN. The U-Net is created by adding down-sampling and up-sampling modules to the left and right of the middle module at each iteration until the input and output modules are reached. The code builds a U-Net with more layers than depicted in the image provided, and it goes 8 layers down, resulting in a 1 by 1 image in the middle of the U-Net, which is then up-sampled to produce a 256 by 256 image with two channels. The author of the paper recommends playing with the code to fully understand its functionality.

4. The architecture of our discriminator is straight forward. This code implements a model by stacking blocks of Conv-BatchNorm-LeackyReLU to decide whether the input image is fake or real. Notice that the first and last blocks do not use normalization and the last block has no activation function. In a "patch" discriminator, the model outputs a number for each patch of the input image, rather than just one number for the entire image. This allows the model to make local changes and decide whether each patch is real or fake separately. The output shape of the model is 30 by 30, but the actual patch size is 70 by 70, which is computed by taking the receptive field of each of the 900 output numbers. Using a patch discriminator for colorization tasks seems reasonable because local changes are important and a vanilla discriminator may not be able to capture these subtleties.

5. In a "patch" discriminator, the model outputs a number for each patch of the input image, rather than just one number for the entire image. This allows the model to make local changes and decide whether each patch is real or fake separately. The output shape of the model is 30 by 30, but the actual patch size is 70 by 70, which is computed by taking the receptive field of each of the 900 output numbers. Using a patch discriminator for colorization tasks seems reasonable because local changes are important and a vanilla discriminator may not be able to capture these subtleties.

6. I initialized the weights of the model with a mean of 0.0 and standard deviation of 0.02 which are the proposed hyperparameters in the paper.

Cheers :)



