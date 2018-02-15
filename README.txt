This is a Beta version for the code of our CVPR 2017 paper:  Efficient Multiple Instance Metric Learning using Weakly Supervised Data

The MATLAB script demo.m in the UCI_Corel5K folder performs both training and test on the UCI Corel5K dataset.
We tested it on 3 different computers and obtained slightly different results on each machine. This may be due to different implementations of SVD and/or pseudoinverse. We then also provide the metric matrix we used for the paper (set "training = false" in demo.m to use it).

We also provide the code to train and test our different models on the splits used for the Label Yahoo! News dataset in the LabelYahooNews folder. The dataset features available at https://lear.inrialpes.fr/people/guillaumin/lyn_train-test.mat have to be included.


Marc T. Law
