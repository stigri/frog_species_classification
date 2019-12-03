## script to start instance on https://www.packet.com/ and to install software needed for training
#!/bin/bash
##
### Packet.net
## apt-get install golang-go
## go get -u github.com/ebsarr/packet # as user installs to ~/go/bin
##
## packet.net admin add-profile
## packet.net baremetal create-device --spot-instance --spot-price-max 0.25 --hostname tf.griep.at --os-type ubuntu_18_04 --facility dfw2 --project-id d673cafc-0f2e-4669-b5b4-7d6627f653fe --plan 18810cec-1711-4f30-b323-a1001804b10f --userfile provision.sh
##
###############################
###############################
##
## Indicate script ran (mkdir atomically).
mkdir /tmp/data-file-init-start/

## This is an unattended install. Nobody is there to press any buttons
export DEBIAN_FRONTEND=noninteractive

## Create User and allow become root
useradd --home /home/stine -m stine -s /bin/bash -G sudo
echo "stine ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

## ssh
mkdir -p /home/stine/.ssh
echo "ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEAvHdZH+pPrDR7tTNVtQPO0GZHsEt43RFWRzvEqkQsub7/s2n9ASwDAkUm+gyvEEH1gGvCVhUkplqkLhw9dZexYDQPSSzeJ7UAGT4zUdJdESeuZdG2+PGO/qY51q6GhO902a+uEN/Ea+IHGQvPW+U9np7joU/jC2OeL53/mO0tWEgeo6fefFhayMKAvuYHj5wDwMjb9Zrlw+7Vdx/n4A9emgPeB57Yg/DDPNjEvoKm+bZdhnrFIKEzNOMEe/Z8nfz9VnE9LpZ0zkBp69zVwsSJEgdHGg7EAiw61djDVGTvlifV9KRDSkXa28RTWYJCAPUCJjGu4zcSV+P+EKlb/D+9Aw== msc-bioinf2019@griep.at" > /home/stine/.ssh/authorized_keys
echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDaFesWicuMjw2s9+4rl36IP781nZ07Vasir5nybvVmmN2wDV1sTcv0iS8VgH54qxmCtV2zQiub0gMq4kHnnTVMKdlMGOyhbvC4X3UVmhJFrD+8UG5bmsbEVXjmgh7Y1oEoldrIf4DlnnHcetdSwuMvV5xqI3iZ+8xg0j9pnN8a9xWj5dUv/rkq2Z5So7AYd+aVCU6uETh8N9fsMZSo/Eu9A+vYvwWhsysY0S8m7wr9zkd71fjc1mTPlAsZGtzACRswrk3S2NLdCd7NNOU1jT5QVffc7poTeCngMFrXjmtUPZZQxOfA6oDq0rSCep+TgjVa2KQAypMDQTjKfkwjaklL markus@kuppe.org" >> /home/stine/.ssh/authorized_keys

## Fix permission because steps are executed by root
chown -R stine:stine /home/stine/

mkdir /tmp/data-file-init-user/

#####################################

## x2go repository
add-apt-repository ppa:x2go/stable -y

mkdir /tmp/data-file-init-add-apt/

#####################################

### https://www.tensorflow.org/install/gpu

# Add NVIDIA package repository
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
echo 'deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /' >> /etc/apt/sources.list.d/nvidia.list
echo 'deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /' >> /etc/apt/sources.list.d/nvidia.list
apt-get update

# Install CUDA and tools. Include optional NCCL 2.x
apt-get install cuda10.0 cuda-cublas-10-0 cuda-cufft-10-0 cuda-curand-10-0 cuda-cusolver-10-0 cuda-cusparse-10-0 cuda-command-line-tools-10-0 libcudnn7=7.4.2.24-1+cuda10.0 libnccl2=2.3.7-1+cuda10.0 -y

mkdir /tmp/data-file-init-cuda/

#####################################

sudo apt-get install --no-install-recommends snapd mate-desktop-environment-extras x2gomatebindings x2goserver x2goserver-xsession x2goserver-extensions sshfs mc zip unzip git git-lfs htop numactl screen python3-opencv python3-matplotlib python3-tk python3-dev python3-pip python3-setuptools -y

mkdir /tmp/data-file-init-install/

#####################################

locale-gen de en

mkdir /tmp/data-file-init-locale/

#####################################

### https://www.tensorflow.org/install/pip
#pip3 install --upgrade pip
#pip3 install --upgrade numpy # Wanted by tf-nightly-gpu, Ubuntu's python3-numpy too old.
pip3 install --upgrade tf-nightly-gpu==1.13.0.dev20190118 # tensorflow-gpu (stable) does not work with cuda 10.

python3 -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))" > /tmp/verify-tf.txt

mkdir /tmp/data-file-init-tensorflow/

#####################################

pip3 install --upgrade keras  # https://keras.io/#installation
pip3 install --upgrade keras-vis
pip3 install --upgrade scikit-learn
pip3 install --upgrade imbalanced-learn

mkdir /tmp/data-file-init-pip-python/

#####################################

# Configure git and clone Stine's git repository
cd /home/stine/
sudo -u stine git config --global user.email "msc-bioinf2019@griep.at"
sudo -u stine git config --global user.name "Stine Griep"
sudo -u stine git clone https://git.informatik.uni-hamburg.de/3griep/msccode.git
cd /home/stine/msccode
sudo -u stine git lfs install
sudo -u stine git lfs pull

mkdir /tmp/data-file-init-git-clone/

#####################################

sudo -u stine echo "termcapinfo xterm* ti@:te@" > /home/stine/.screenrc

#####################################

## install pycharm-community IDE
sudo snap install pycharm-community --classic

## Lastly, mark the completion of this script
touch /home/stine/.cloud-warnings.skip
mkdir /tmp/data-file-init-complete/

#### Check GPU load with nvidia-smi


