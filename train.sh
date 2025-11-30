#!/bin/bash
nohup python train.py --input_option 4x --exp_name share_decoder_6layer_4x_train > train.log 2>&1 &