This is a Beta version for the code of our CVPR 2017 paper:  Efficient Multiple Instance Metric Learning using Weakly Supervised Data

The MATLAB script demo.m gives an instance of how this code is meant to be used, on the UCI Corel5K dataset.
We tested it on 3 different computers and obtained slightly different results on each machine. This may be due to different implementations of SVD and/or pseudoinverse. We then also provide the metric matrix we used for the paper (set "training = false" in demo.m to use it).

We also provide the different splits/folds used for the Label Yahoo! News dataset. They are sufficient to reproduce the results in the paper; it is worth noting that we mean centered the training data for each split.
We plan to release a clean code to reproduce experiments on this dataset in the coming months (probably by July).


Marc T. Law
