# Universal-sketch-perceptual-grouping
This is the training code and model of "Universal Sketch Perceptual grouping ", which was published in ECCV2018
if you use this code, please cited
@article{
  Li2018Universal,
  title={Universal Perceptual Grouping},
  author={Li, Ke and Pang, Kaiyue and Song, Jifei and Song, Yi Zhe and Xiang, Tao and Hospedales, Timothy M and Zhang, Honggang},
  booktitle={ECCV},
  year={2018},
}

You can also find the SketctX-PRIS dataset in 
https://github.com/KeLi-SketchX/SketchX-PRIS-Dataset

This framework is based on sketchrnn, so you should install magenta, see more detail in https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn

Our code contain 3 folds:
1. Item 1 Sketch Perceptual grouper training and inference code--PG_RNN_train.py
If you want to retrain the model or get the affinity metrix \hat{G}, you can use this file.
1. Item 2 The postprocess code of Sketch Perceptual grouping--/ECCV_JULE/test_PG_cluster.py
If you want to get the group, you can use those file.Note: Those file will inference and cluster the affinity metrix \hat{G}.
1. Item 3 The Evaluation code, We use the benchmark in BSDS500. We process our annotation data to get the GrourdTruth.
