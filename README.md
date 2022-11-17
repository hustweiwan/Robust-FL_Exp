# [IEEE MSN 2022] Shielding Federated Learning: Mitigating Byzantine Attacks with Less Constraints
This repository comprises of implementation of Robust-FL (https://arxiv.org/pdf/2210.01437.pdf) under sign flipping attack on MNIST.

## Abstract
Federated learning is a newly emerging distributed learning framework that facilitates the collaborative training of a shared global model among distributed participants with their privacy preserved. However, federated learning systems are vulnerable to Byzantine attacks from malicious participants, who can upload carefully crafted local model updates to degrade the quality of the global model and even leave a backdoor. While this problem has received significant attention recently, current defensive schemes heavily rely on various assumptions, such as a fixed Byzantine model, availability of participants' local data, minority attackers, IID data distribution, etc. 

To relax those constraints, this paper presents Robust-FL, the first prediction-based Byzantine-robust federated learning scheme where none of the assumptions is leveraged. The core idea of the Robust-FL is exploiting historical global model to construct an estimator based on which the local models will be filtered through similarity detection. We then cluster local models to adaptively adjust the acceptable differences between the local models and the estimator such that Byzantine users can be identified. Extensive experiments over different datasets show that our approach achieves the following advantages simultaneously: (i) independence of participants' local data, (ii) tolerance of majority attackers, (iii) generalization to variable Byzantine model.

## Experimental results
![image](https://user-images.githubusercontent.com/102348359/202386307-bcef032b-4ddb-4417-a5c9-d10299b46535.png)


## Citation
```
@article{li2022shielding,
  title={Shielding Federated Learning: Mitigating Byzantine Attacks with Less Constraints},
  author={Li, Minghui and Wan, Wei and Lu, Jianrong and Hu, Shengshan and Shi, Junyu and Zhang, Leo Yu},
  journal={arXiv preprint arXiv:2210.01437},
  year={2022}
}
```
