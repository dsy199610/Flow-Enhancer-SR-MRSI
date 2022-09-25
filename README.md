# Flow-based Visual Quality Enhancer for Super-resolution Magnetic Resonance Spectroscopic Imaging (DGM4MICCAI 2022)

Siyuan Dong, Gilbert Hangel, Eric Z Chen, Shanhui Sun, Wolfgang Bogner, Georg Widhalm, Chenyu You, John A Onofrey, Robin de Graaf, James S Duncan

[[Paper Link](https://arxiv.org/abs/2207.10181)]

### Citation
If you use this code please cite:

    @article{dong2022flow,
      title={Flow-based Visual Quality Enhancer for Super-resolution Magnetic Resonance Spectroscopic Imaging},
      author={Dong, Siyuan and Hangel, Gilbert and Chen, Eric Z and Sun, Shanhui and Bogner, Wolfgang and Widhalm, Georg and You, Chenyu and Onofrey, John A and de Graaf, Robin and Duncan, James S},
      journal={arXiv preprint arXiv:2207.10181},
      year={2022}
    }
   
### Environment and Dependencies
 Requirements:
 * python 3.7.11
 * pytorch 1.1.0
 * pytorch-msssim 0.2.1
 * torchvision 0.3.0
 * numpy 1.19.2

### Directory
    main.py                             # main file for flow-based enhancer networks
    main_MUNet.py                       # main file for super-resolution network MUNet
    main_cWGAN.py                       # main file for MUNet-cWGAN
    loader
    └──  dataloader.py                  # dataloader
    utils
    ├──logs.py                          # logging
    └──utils.py                         # utility files
    models
    ├──MUNet.py                         # Super-resolution network
    ├──cInvNet.py                       # baseline flow-based network (SRFlow)
    ├──cInvNet_ConditionalPrior.py      # Abalation study - no MRI priors
    ├──cInvNet_MRI_LearnablePrior.py    # Abalation study - no conditional base distribution
    ├──cInvNet_MRI_ConditionalPrior.py  # Our model
    └──cWGAN.py                         # functions for training cWGAN

