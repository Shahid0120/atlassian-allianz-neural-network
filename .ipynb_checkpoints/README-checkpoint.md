# Atlassian Allianz Neural Network

In this notebook, I will extend the work from the Atlassian Allianz Data Competition. 

One of the main criticisms of our solution was that we discussed why we didn't use neural networks but never showed any implementation to compare performance. Additionally, another criticism was that we never showed any experimentation comparing performance on the raw dataset versus iterations of processed data. 

In an effort to rectify this, I will implement different architectures of neural networks and compare performance on raw and iterated processed data.


# Interpretability 
A major component of presenting Neural Network is the ability for non-technical managers to understand Neural Network outputs. In this papers i implemented 3 main strategies

1. Distillation Method (Local Approximation) - Anchors: High-Precision Model-Agnostic Explanations by Ribeiro, M. T., Singh, S., and Guestrin, C. (2018)
2. Distillation (Model translation) - Tree Based: Learning global additive explanations for neural nets using model distillation by Tan, S., Caruana, R., Hooker, G., Koch, P., and Gordo, A. (2018)
3. Intrinsic Joint Training (Explanation Association): Towards robust interpretability with self-explaining neural networks by Melis, D. A. and Jaakkola, T. (2018).

All these method where categories according Explainable Deep Learning: A Field Guide for the Uninitiated by Gabriëlle Ras, Ning Xie, Marcel van Gerven, Derek Doran