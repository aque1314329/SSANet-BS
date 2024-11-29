# SSANet-BS
SSANet-BS: Spectral–Spatial Cross-Dimensional Attention Network for Hyperspectral Band Selection

https://www.mdpi.com/2072-4292/16/15/2848

# Abstract
Band selection (BS) aims to reduce redundancy in hyperspectral imagery (HSI). Existing BS approaches typically model HSI only in a single dimension, either spectral or spatial, without exploring the interactions between different dimensions. To this end, we propose an unsupervised BS method based on a spectral–spatial cross-dimensional attention network, named SSANet-BS. This network is comprised of three stages: a band attention module (BAM) that employs an attention mechanism to adaptively identify and select highly significant bands; two parallel spectral–spatial attention modules (SSAMs), which fuse complex spectral–spatial structural information across dimensions in HSI; a multi-scale reconstruction network that learns spectral–spatial nonlinear dependencies in the SSAM-fusion image at various scales and guides the BAM weights to automatically converge to the target bands via backpropagation. The three-stage structure of SSANet-BS enables the BAM weights to fully represent the saliency of the bands, thereby valuable bands are obtained automatically. Experimental results on four real hyperspectral datasets demonstrate the effectiveness of SSANet-BS.

# How to run
Install dependency: tqdm, torch and scipy.
Set your dataset, nbs (number of bands to select) and hyper parameter in main.py.
Run python main.py, the selected bands will display in terminal.
