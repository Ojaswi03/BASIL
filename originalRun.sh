#!/bin/bash

echo "🔁 Starting Basil Experiments..."

# Core parameters that main.py expects
dataset="m"
nodes=10
S=10
rounds=20
epochs=50
iid="y"
acds="n"
basilPlus="n"
attack=["none" "gaussian" "sign_flip" "hidden"]
attkID=[2,5,7]

echo "======================= CONFIG ================="
echo "Dataset               : $dataset"
echo "Nodes                 : $nodes"
echo "S                     : $S"
echo "Rounds                : $rounds"
echo "Epochs                : $epochs"
echo "IID [y/n]             : $iid"
echo "ACDS [y/n]            : $acds"
echo "Basil Plus [y/n]      : $basilPlus"
echo "Attack                : $attack"
echo "Attack IDs (Nodes)    : $attkID"
echo "================================================="


# Run MNIST
echo "🧠 Running Basil with MNIST..."
python3 main_basil.py <<EOF
dataset
nodes
S
rounds
epochs
iid
acds
basilPlus
attack
attkID
EOF



# TESTING CONFIGURATION FOR CIFAR-10
dataset="c"
nodes=10
S=10
rounds=20
epochs=1
iid="y"
acds="n"
basilPlus="n"
attacks=("none" "gaussian" "sign_flip" "hidden")
attkID="2,5,7"
echo "======================= CONFIG ================="
echo "Dataset               : $dataset"
echo "Nodes                 : $nodes"
echo "S                     : $S"
echo "Rounds                : $rounds"
echo "Epochs                : $epochs"
echo "IID [y/n]             : $iid"
echo "ACDS [y/n]            : $acds"
echo "Basil Plus [y/n]      : $basilPlus"
echo "Attack                : $attack"
echo "Attack IDs (Nodes)    : $attkID"
echo "================================================="


# Run CIFAR-10
echo "🖼️  Running Basil with CIFAR-10..."
python3 main_basil.py <<EOF
$dataset
$nodes
$S
$rounds
$epochs
$iid
$acds
$basilPlus
$attack
$attkID
EOF

echo "✅ All experiments completed."