#!/bin/bash

echo "🔁 Starting Basil Experiments..."

# ==== CONFIG ====
dataset="c"        # "m" for MNIST, "c" for CIFAR10
nodes=10
S=3
rounds=500
epochs=1
iid="y"            # "y" for IID, "n" for non-IID
acds="y"           # only used if iid="n"
basilPlus="n"      # "y" to enable Basil+
attack="hidden"    # "none", "gaussian", "sign_flip", "hidden"
attackers="2,5,7"
hiddenStart=20

# ==== PRINT CONFIG ====
echo ""
echo "======================= CONFIG ================="
echo "Dataset               : $dataset"
echo "Nodes                 : $nodes"
echo "S                     : $S"
echo "Rounds                : $rounds"
echo "Epochs                : $epochs"
echo "IID [y/n]             : $iid"
if [ "$iid" == "n" ]; then
  echo "ACDS [y/n]            : $acds"
fi
echo "Basil Plus [y/n]      : $basilPlus"
echo "Attack Type           : $attack"
echo "Attacker IDs          : $attackers"
if [ "$attack" == "hidden" ]; then
  echo "Hidden Start Round    : $hiddenStart"
fi
echo "================================================="
echo ""

# ==== RUN PYTHON SCRIPT WITH CORRECT INPUTS ====
if [ "$iid" == "y" ]; then
python3 main_basil.py <<EOF
$dataset
$nodes
$S
$rounds
$epochs
$iid
$basilPlus
$attack
$attackers
$hiddenStart
EOF
else
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
$attackers
$hiddenStart
EOF
fi

echo "✅ All experiments completed."
