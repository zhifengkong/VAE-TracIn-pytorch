# Official PyTorch implementation for "Understanding Instance-based Interpretability of Variational Auto-Encoders".

This repo implements VAE-TracIn, an instance based interpretation method for variational autoencoders. See paper via [this link](https://arxiv.org/pdf/2105.14203.pdf).

# Train VAE Models
To train VAE models, ```cd ./Beta_VAE/```, modify dataset paths in ```paths.py```, and then run
- MNIST: ```python main.py --dataset mnist --seed 1 --optim sgd --lr 1e-4 --objective H --model MNIST --batch_size 64 --z_dim 128 --max_iter 1.5e6 --beta 4 --viz_on False --viz_name mnist_beta4_z128_sgd```
- CIFAR subclass: ```python main.py --dataset cifar_zero --seed 1 --optim sgd --lr 3e-4 --objective H --model H --batch_size 64 --z_dim 64 --max_iter 1.5e6 --beta 2 --viz_on False --viz_name cifar_zero_beta2_z64_sgd```
- CIFAR: ```python main.py --dataset cifar --seed 1 --optim sgd --lr 3e-4 --objective H --model H --batch_size 64 --z_dim 128 --max_iter 1.5e6 --beta 2 --viz_on False --viz_name cifar_beta2_z128_sgd```
- MNIST data cleaning: ```python main.py --dataset mnist_emnist_extra --seed 1 --optim sgd --lr 1e-4 --objective H --model MNIST --batch_size 64 --z_dim 128 --max_iter 1.5e6 --beta 4 --viz_on False --viz_name mnist_emnist_extra_beta4_z128_sgd```
- CIFAR data cleaning: ```python main.py --dataset cifar_celeba_extra --seed 1 --optim sgd --lr 3e-4 --objective H --model H --batch_size 64 --z_dim 128 --max_iter 1.5e6 --beta 2 --viz_on False --viz_name cifar_celeba_extra_beta2_z128_sgd```
  
Note: in CIFAR subclass experiments, the dataset parameter can be cifar_zero through cifar_nine. **The cifar_zero pretrained models are provided.**


# Sanity Check
- MNIST: ```python VAE_TracIn_testinf.py --beta 4 --latent_dim 128 --dataset mnist --task train```
- CIFAR: ```python VAE_TracIn_testinf.py --beta 2 --latent_dim 128 --dataset cifar --task train```
- CIFAR subclass: ```python VAE_TracIn_testinf.py --beta 2 --latent_dim 64 --dataset cifar_zero --task train```


# Self influences
- MNIST: ```python VAE_TracIn_selfinf.py --beta 4 --latent_dim 128 --dataset mnist --last_epoch```
- CIFAR: ```python VAE_TracIn_selfinf.py --beta 2 --latent_dim 128 --dataset cifar --last_epoch```
- CIFAR subclass: ```python VAE_TracIn_selfinf.py --beta 2 --latent_dim 64 --dataset cifar_zero --last_epoch```
- MNIST data cleaning: ```python VAE_TracIn_selfinf.py --beta 4 --latent_dim 128 --dataset mnist_emnist_extra --last_epoch```
- CIFAR data cleaning: ```python VAE_TracIn_selfinf.py --beta 2 --latent_dim 128 --dataset cifar_celeba_extra --last_epoch```


# Influences over test data
- MNIST: ```python VAE_TracIn_testinf.py --beta 4 --latent_dim 128 --dataset mnist --task test --load_from_jpg --load_from_jpg_dir output/images/test_mnist```
- CIFAR subclass: ```python VAE_TracIn_testinf.py --beta 2 --latent_dim 128 --dataset cifar --task test --load_from_jpg --load_from_jpg_dir output/images/test_cifar```
- CIFAR subclass: ```python VAE_TracIn_testinf.py --beta 2 --latent_dim 64 --dataset cifar_zero --task test --load_from_jpg --load_from_jpg_dir output/images/test_cifar_zero```


# References
- [Beta VAE](https://github.com/1Konny/Beta-VAE)
- [Influence Functions PyTorch](https://github.com/nimarb/pytorch_influence_functions)
- [TracIn TensorFlow](https://github.com/frederick0329/TracIn)
- [Data Copying](https://github.com/casey-meehan/data-copying)
