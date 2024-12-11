# Detecting and Corrupting Convolution-based Unlearnable Examples
The official implementation of our AAAI 2025 paper "*[Detecting and Corrupting Convolution-based Unlearnable Examples](https://arxiv.org/pdf/2311.18403)*", by *[Minghui Li*](http://trustai.cse.hust.edu.cn/index.htm), [Xianlong Wang*✉](https://wxldragon.github.io/), Zhifei Yu, [Shengshan Hu](http://trustai.cse.hust.edu.cn/index.htm), [Ziqi Zhou](https://zhou-zi7.github.io/), [Longling Zhang](https://scholar.google.com.hk/citations?user=3YvpfSwAAAAJ&hl=zh-CN&oi=ao), and [Leo Yu Zhang](https://scholar.google.com.hk/citations?user=JK21OM0AAAAJ&hl=zh-CN&oi=ao).*

![AAAI 2025](https://img.shields.io/badge/AAAI-2025-blue.svg?style=plastic) 
![Unlearnable Examples](https://img.shields.io/badge/Unlearnable-Examples-yellow.svg?style=plastic)
![Convolutional Noise](https://img.shields.io/badge/Convolutional-Noise-orange.svg?style=plastic)

## Abstract
Convolution-based unlearnable examples (UEs) employ class-wise multiplicative convolutional noise to training samples, severely compromising model performance. This firenew type of UEs have successfully countered all defense mechanisms against UEs. The failure of such defenses can be
attributed to the absence of norm constraints on convolutional noise, leading to severe blurring of image features. To address this, we first design an Edge Pixel-based Detector (EPD) to identify convolution-based UEs. Upon detection of them, we propose the first defense scheme against convolution-based UEs, COrrupting these samples via random matrix multiplication by employing bilinear INterpolation (COIN) such that
disrupting the distribution of class-wise multiplicative noise. To evaluate the generalization of our proposed COIN, we newly design two convolution-based UEs called VUDA and HUDA to expand the scope of convolution-based UEs. Extensive experiments demonstrate the effectiveness of detection scheme EPD and that our defense COIN outperforms 11 state-of-the-art (SOTA) defenses, achieving a significant improvement on the CIFAR and ImageNet datasets.
