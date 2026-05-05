#!/bin/bash
# SwiftCAD paper main config (Table 2 row "Shared + w/o MLP (144)").
# Defaults already enable shared decoder and disable MLP embedding;
# d_model is set explicitly for clarity.
nohup python train.py --input_option 4x --d_model 144 --exp_name swiftcad_main > train.log 2>&1 &
