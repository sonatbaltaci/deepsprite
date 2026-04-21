set -e
mkdir -p datasets

# affNIST
wget 'https://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/test.mat.zip' --output-document test.mat.zip
unzip test.mat.zip && rm test.mat.zip
mv test.mat datasets/affNIST_test.mat

# Multi-object datasets
gdown --id 1IQ8Uk7zs4fJSjAG70HgecbwCzKxgTJ-U -O tetrominoes.zip
unzip tetrominoes.zip && rm tetrominoes.zip
mv tetrominoes datasets/
gdown --id 1F-imgv_wj236XVj6wWpDdPouHJtZGWkT -O dsprites_gray.zip
unzip dsprites_gray.zip && rm dsprites_gray.zip
mv dsprites_gray datasets/
gdown --id 15lmua6k2RYlWDuHHrQjtwcMIDzbtvR4F -O clevr6.zip
unzip clevr6.zip && rm clevr6.zip
mv clevr6 datasets/
gdown --id 1Cepxzc5veuA43soRt5lwG15b7srEt1l5 -O clevr.zip
unzip clevr.zip && rm clevr.zip
mv clevr datasets/

# GTSRB
gdown --id 1FyVZVtjz2auFBR-O4X2MA7vfjT3evDuW -O GTSRB.zip
unzip GTSRB.zip && rm GTSRB.zip
mv GTSRB datasets/