curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ./Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -p ./miniconda3 -b
source miniconda3/bin/activate
conda create -n torchbend python=3.11 -y

conda activate torchbend
pip install ".[$(IFS=,; echo "$*")]"