#!/bin/bash

echo "🔁 Starting Basil Experiments..."

dataset="m"
nodes=10
S=10
rounds=35
epochs=1
iid="y"
acds="n"
basilPlus="n"

echo ""
echo "======================= CONFIG ================="
echo "Dataset               : $dataset"
echo "Nodes                 : $nodes"
echo "S                     : $S"
echo "Rounds                : $rounds"
echo "Epochs                : $epochs"
echo "IID [y/n]             : $iid"
echo "ACDS [y/n]            : $acds"
echo "Basil Plus [y/n]      : $basilPlus"
echo "================================================="


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

echo "All experiments completed."
