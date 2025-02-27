# DeepCTR-Torch

[![Python Versions](https://img.shields.io/pypi/pyversions/deepctr-torch.svg)](https://pypi.org/project/deepctr-torch)
[![Downloads](https://pepy.tech/badge/deepctr-torch)](https://pepy.tech/project/deepctr-torch)
[![PyPI Version](https://img.shields.io/pypi/v/deepctr-torch.svg)](https://pypi.org/project/deepctr-torch)
[![GitHub Issues](https://img.shields.io/github/issues/shenweichen/deepctr-torch.svg
)](https://github.com/shenweichen/deepctr-torch/issues)


[![Documentation Status](https://readthedocs.org/projects/deepctr-torch/badge/?version=latest)](https://deepctr-torch.readthedocs.io/)
![CI status](https://github.com/shenweichen/deepctr-torch/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/shenweichen/DeepCTR-Torch/branch/master/graph/badge.svg)](https://codecov.io/gh/shenweichen/DeepCTR-Torch)
[![Gitter](https://badges.gitter.im/DeepCTR/community.svg)](https://gitter.im/DeepCTR/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![License](https://img.shields.io/github/license/shenweichen/deepctr-torch.svg)](https://github.com/shenweichen/deepctr-torch/blob/master/LICENSE)

PyTorch version of [DeepCTR](https://github.com/shenweichen/DeepCTR).

DeepCTR is a **Easy-to-use**,**Modular** and **Extendible** package of deep-learning based CTR models along with lots of core components layers which can be used to build your own custom model easily.You can use any complex model with `model.fit()`and `model.predict()` .Install through `pip install -U deepctr-torch`.

Let's [**Get Started!**](https://deepctr-torch.readthedocs.io/en/latest/Quick-Start.html)([Chinese Introduction](https://zhuanlan.zhihu.com/p/53231955))

## Contributors([welcome to join us!](./CONTRIBUTING.md))
<a href="https://github.com/shenweichen">
    <img src="https://avatars.githubusercontent.com/shenweichen " width=70 height="70" alt="pic" >
</a>
<a href="https://github.com/wutongzhang">
    <img src="https://avatars.githubusercontent.com/wutongzhang " width=70 height="70" alt="pic" >
</a>
<a href="https://github.com/JyiHUO">
    <img src="https://avatars.githubusercontent.com/JyiHUO " width=70 height="70" alt="pic" >
</a>
<a href="https://github.com/chenkkkk">
    <img src="https://avatars.githubusercontent.com/chenkkkk " width=70 height="70" alt="pic" >
</a>
<a href="https://github.com/tangaqi">
    <img src="https://avatars.githubusercontent.com/tangaqi " width=70 height=70" alt="pic" >
</a>
<a href="https://github.com/uestc7d">
    <img src="https://avatars.githubusercontent.com/uestc7d " width=70 height="70" alt="pic" >
</a>


## Models List

|                 Model                  | Paper                                                                                                                                                           |
| :------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Factorization-supported Neural Network | [ECIR 2016][Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf)                    |
|      Product-based Neural Network      | [ICDM 2016][Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                   |
|              Wide & Deep               | [DLRS 2016][Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                                 |
|                 DeepFM                 | [IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)                           |
|        Piece-wise Linear Model         | [arxiv 2017][Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/abs/1704.05194)                                 |
|          Deep & Cross Network          | [ADKDD 2017][Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)                                                                   |
|   Attentional Factorization Machine    | [IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435) |
|      Neural Factorization Machine      | [SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)                                               |
|                xDeepFM                 | [KDD 2018][xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                         |
|                AutoInt                 | [arxiv 2018][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                              |
|                  ONN      | [arxiv 2019][Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf)                                                           |
|                  FiBiNET               | [RecSys 2019][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)   |

