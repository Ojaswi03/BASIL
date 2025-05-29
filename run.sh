#!/bin/bash

echo "🔁 Starting Basil Experiments..."

# Base config
# datasets=("m" "c")
# attacks=("none" "gaussian" "sign_flip" "hidden")
# nodes=10
# S=10
# rounds=20
# epochs=50
# iid="y"
# acds="n"
# basilPlus="n"
# attkID="2,5,7"

# for dataset in "${datasets[@]}"; do
#     for attack in "${attacks[@]}"; do
#         echo "======================= CONFIG ================="
#         echo "Dataset               : $dataset"
#         echo "Nodes                 : $nodes"
#         echo "S                     : $S"
#         echo "Rounds                : $rounds"
#         echo "Epochs                : $epochs"
#         echo "IID [y/n]             : $iid"
#         echo "ACDS [y/n]            : $acds"
#         echo "Basil Plus [y/n]      : $basilPlus"
#         echo "Attack                : $attack"
#         echo "Attack IDs (Nodes)    : $attkID"
#         echo "================================================="

#         python3 main_basil.py <<EOF
# $dataset
# $nodes
# $S
# $rounds
# $epochs
# $iid
# $acds
# $basilPlus
# $attack
# $attkID
# EOF
#     done
# done




#Testing configuration for CIFAR-10


dataset='c'
nodes=10
S=10
rounds=500
epochs=1

iid='y'
acds='n'
basilPlus='n'

# attack='gaussian'
# attack_ids='2,5,7'


# # Include 'none' as a valid attack type for completeness
# # declare -a attacks=("gaussian" "sign_flip" "hidden")
# # list of attacks to test
# attacks=("none" "gaussian" "sign_flip" "hidden")
# attacks=("gaussian")

# for attack in "${attacks[@]}"; do
#     echo ""
#     echo "======================= CONFIG ================="
#     echo "Dataset               : $dataset"
#     echo "Nodes                 : $nodes"
#     echo "S                     : $S"
#     echo "Rounds                : $rounds"
#     echo "Epochs                : $epochs"
#     echo "IID [y/n]             : $iid"
#     echo "ACDS [y/n]            : $acds"
#     echo "Basil Plus [y/n]      : $basilPlus"
#     echo "Attack                : $attack"
#     echo "Attack IDs (Nodes)    : $attack_ids"
#     # if [ "$attack" = "none" ]; then
#     #     echo "Attack IDs (Nodes)    :  None (No attack)"""
#     # else
#     #     echo "Attack IDs (Nodes)    : $attack_ids"
#     # fi
#     echo "================================================="

#     # Launch experiment


python3 main_basil.py <<EOF
$dataset
$nodes
$S
$rounds
$epochs
$iid
$acds
$basilPlus
EOF
# done

echo "✅ All experiments completed."
##$( [ "$attack" = "none" ] && echo "" || echo "$attack_ids" )