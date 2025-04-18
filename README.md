# DHNet
## Introduction
This is our implementation of our paper *Debiased Hypernetwork Is A Generated Class-Incremental Learner*.

**TL;DR**: A generated adapter based method for class-incremental learning.

**Abstract**:
Class-incremental learning (Class-IL) imposes a significant challenge on the catastrophic forgetting. Although parameter isolation-based methods have shown a great potential in handling catastrophic forgetting by expanding sub-network branches for streaming tasks, such expansion strategy generally leads to a dramatic increase of network size. By observing that, adapters can fit different tasks by adding a small-size model within a large-size network, whereas transformers can act as adapters whose parameters are suitable to be generated by hypernetworks, we propose a Debiased Hypernetwork (DHNet) for Class-IL by incorporating hypernetwork, adapter, and transformer together for Class-IL. In details, DHNet consists of an adapter accumulation stage and a debiased distillation stage. The first stage selects to generate the last transformer layer (known as the adapter) with a hypernetwork to effectively transplant catastrophic forgetting to the hypernetwork, where the adapter is built with spatial and channel chunking strategies.
The second stage mitigates the catastrophic forgetting of the hypernetwork by co-training the hypernetwork and task-conditioners together, which can effectively reduce the hypernetwork biase induced from distillation process and facilitate long-term knowledge retention. Thanks to the cooperation of chunking strategies on adapters and co-training strategy on hypernetwork, DHNet achieves satisfactory compromise between the parameter size and anti-forgetting performance. Results Comprehensive experiments on CIFAR-100 and ImageNet datasets demonstrate the superiority of  DHNet over existing Class-IL methods. 


## Dependencies
- numpy==1.23.5
- torch==1.12.1
- torchvision==0.13.1
- timm==0.4.9
- continuum==1.2.7 



## Usage
##### 1.Run code
For dataset CIFAR100 as an example
```
bash train.sh 0,1   --options options/data/cifar100_10-10.yaml  options/data/cifar100_order1.yaml  options/model/cifar_hyper_gan.yaml     --name Hyper_GAN_Vit     --data-path /dataset/    --output-basedir /checkpoint   --memory-size 2000
```
