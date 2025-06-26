FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    cmake \
    build-essential \
    libopencv-dev \
    python3-opencv \
    ros-noetic-desktop-full \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

WORKDIR /workspace

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/
COPY msg/ ./msg/
COPY package.xml .
COPY CMakeLists.txt .

FROM base as vla-enhanced

RUN pip3 install --no-cache-dir \
    transformers[torch] \
    accelerate \
    peft \
    bitsandbytes \
    datasets

ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

EXPOSE 11311

CMD ["bash", "-c", "source /opt/ros/noetic/setup.bash && python3 scripts/leaf_grasp_node_vla.py"] 