#!/bin/bash

python3 FeatureExtractionCNN.py --data_dir "results_problem_1/" --model_name "p1_cnn_5e" --loss_file "p1_cnn_loss_5e"
python3 FeatureExtractionCNN.py --data_dir "results_problem_20/" --model_name "p20_cnn_5e" --loss_file "p20_cnn_loss_5e"
python3 FeatureExtractionCNN.py --data_dir "results_problem_21/" --model_name "p21_cnn_5e" --loss_file "p21_cnn_loss_5e"
python3 FeatureExtractionCNN.py --data_dir "results_problem_5/" --model_name "p5_cnn_5e" --loss_file "p5_cnn_loss_5e"