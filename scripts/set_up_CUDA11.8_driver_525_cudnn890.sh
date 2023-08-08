sudo apt update

sudo apt-get update

sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release -y

sudo mkdir -m 0755 -p /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get install libxml2

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

sudo apt install apt-transport-https ca-certificates curl software-properties-common -y

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu jammy stable"

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y

sudo apt-get install libxml2

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

sudo systemctl restart docker

sudo apt install wget

wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

chmod 777 ./cuda_11.8.0_520.61.05_linux.run

sudo apt update

sudo apt install build-essential -y

sudo apt install linux-headers-$(uname -r)

sudo apt install libxml2

sudo apt install libncurses5-dev -y

sudo apt-get install gcc-12=12.1.0-2ubuntu1~22.04

sudo update-alternatives --remove-all gcc

sudo update-alternatives --remove-all cc

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 20

sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30

sudo ./cuda_11.8.0_520.61.05_linux.run --silent --driver --toolkit

wget https://us.download.nvidia.com/XFree86/Linux-x86_64/525.116.03/NVIDIA-Linux-x86_64-525.116.03.run

lspci -vnn | grep VGA 

sudo apt-get remove --purge nvidia* -y

chmod 777 ./NVIDIA-Linux-x86_64-525.116.03.run

sudo apt install apt-transport-https ca-certificates curl software-properties-common -y

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu jammy stable"

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo apt-get install libxml2

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

sudo apt install apt-transport-https ca-certificates curl software-properties-common -y

sudo docker run --shm-size=16G --gpus all --name newcontainer -it -d gcr.io/tpu-pytorch/xla:nightly_3.8_cuda_11.8 bin/bash

sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f

sudo docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f

sudo apt-get purge nvidia-docker

curl https://get.docker.com | sh

sudo systemctl start docker && sudo systemctl enable docker

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

curl -s -L https://nvidia.github.io/nvidia-container-runtime/experimental/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list 

sudo apt-get update

sudo apt-get install -y nvidia-docker2

sudo systemctl restart docker

lspci | grep -i nvidia

gsutil cp gs://yeounoh-shared/cudnn-local-repo-ubuntu2204-8.9.0.131_1.0-1_amd64.deb .

chmod 777 cudnn-local-repo-ubuntu2204-8.9.0.131_1.0-1_amd64.deb

sudo -s

sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.0.131_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.0.131/cudnn-local-D7522631-keyring.gpg /usr/share/keyrings/

sudo apt-get update

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin 

sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get install libcudnn8=8.9.0.131-1+cuda11.8

sudo apt-get install libcudnn8-dev=8.9.0.131-1+cuda11.8

sh NVIDIA-Linux-x86_64-525.116.03.run --no-opengl-files

nvidia-smi -pm 1

sudo docker run --shm-size=16G --gpus all --name netnenewnewnew --network host -it -d gcr.io/tpu-pytorch/xla:nightly_3.8_cuda_11.8 bin/bash

sudo docker ps -a

sudo docker exec -it netnenewnewnew /bin/bash

gsutil cp gs://yeounoh-shared/A3/cudnn-linux-x86_64-8.9.0.131_cuda11-archive.tar.xz .

tar -xvf cudnn-linux-x86_64-8.9.0.131_cuda11-archive.tar.xz

cp /cudnn-linux-x86_64-8.9.0.131_cuda11-archive/include/* /usr/local/cuda/include
cp /cudnn-linux-x86_64-8.9.0.131_cuda11-archive/include/* /usr/include/
cp -P /cudnn-linux-x86_64-8.9.0.131_cuda11-archive/lib/libcudnn* /usr/local/cuda/lib64

sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn* /usr/include/cudnn*.h
