# trustFL
Secure and Private Peer-to-Peer Federated Learning: A Distributed Approach to Collaborative Deep Learning

## Installation
Install all necessary packages using requirements.txt

`pip install -r requirements.txt`

To install PyTorch 1.9.0 with CUDA 11.1 support, you can use the following command:

`pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html`

Download the preprocessed FEMNIST dataset from drive https://drive.google.com/file/d/19YPsKcBoeU_ikHtepwtCm8fi664Cys_h/view?usp=share_link

Extract and copy the pickle_datasets folder to datastets folder
`cp pickle_datasets trustFL\datasets`

Now execute all notebooks or to run our proposed model
`jupyter notebook`
`Run trustfl_femnist_Floppy.ipynb`
