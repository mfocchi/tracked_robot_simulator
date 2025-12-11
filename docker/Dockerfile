# ==========================================================
# Ubuntu 20.04 + ROS Noetic full desktop
# ==========================================================
FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# -----------------------------
# Base system dependencies
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales tzdata curl wget gnupg2 lsb-release ca-certificates \
    software-properties-common git build-essential cmake \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8

# -----------------------------
# Install ROS build tools
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-rosdep python3-vcstool python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Initialize rosdep
# -----------------------------
RUN rosdep init || true && rosdep update

# -----------------------------
# Install useful ROS packages
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-urdfdom-py \
    ros-noetic-srdfdom \
    ros-noetic-joint-state-publisher \
    ros-noetic-joint-state-publisher-gui \
    ros-noetic-joint-state-controller \
    ros-noetic-gazebo-msgs \
    ros-noetic-control-toolbox \
    ros-noetic-gazebo-ros \
    ros-noetic-controller-manager \
    ros-noetic-joint-trajectory-controller \
    ros-noetic-catkin \
    ros-noetic-openni2-launch \
    ros-noetic-openni2-camera \
    ros-noetic-realsense2-description \
    ros-noetic-eigen-conversions \
    ros-noetic-object-recognition-msgs \
    ros-noetic-roslint \
    ros-noetic-ur-msgs \
    ros-noetic-speed-scaling-interface \
    ros-noetic-scaled-joint-trajectory-controller \
    ros-noetic-speed-scaling-state-controller \
    ros-noetic-ur-client-library \
    ros-noetic-pass-through-controllers \
    ros-noetic-gazebo-plugins \
    ros-noetic-gazebo-ros-control \
    ros-noetic-ros-control \
    ros-noetic-transmission-interface \
    ros-noetic-imu-tools \
    ros-noetic-tf2-ros \
    ros-noetic-tf2-tools \
    ros-noetic-industrial-robot-status-controller \
    rospack-tools \ 
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Python 
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-scipy \
    python3-matplotlib \
    python3-termcolor \
    python3-tk \
    python3-catkin-pkg \
    python3-rospkg \   
    python3-rosdep \ 
    && rm -rf /var/lib/apt/lists/*
 
RUN pip3 install --no-cache-dir \  
    cvxpy==1.2.0 \
    shapely \
    quadprog 

# -----------------------------
# robotpkg (Focal → py38)
# -----------------------------
RUN set -eux; \
    install -d -m 0755 /etc/apt/keyrings; \
    curl -fsSL http://robotpkg.openrobots.org/packages/debian/robotpkg.key \
      | gpg --dearmor -o /etc/apt/keyrings/robotpkg-archive-keyring.gpg; \
    chmod 0644 /etc/apt/keyrings/robotpkg-archive-keyring.gpg; \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg-archive-keyring.gpg] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -sc) robotpkg" \
      >> /etc/apt/sources.list.d/robotpkg.list; \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg-archive-keyring.gpg] http://robotpkg.openrobots.org/wip/packages/debian/pub $(lsb_release -sc) robotpkg" \
      >> /etc/apt/sources.list.d/robotpkg.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    robotpkg-py38-eigenpy \
    robotpkg-py38-pinocchio \
    robotpkg-py38-quadprog \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Utilities and GUI dependencies
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gedit gitg git-cola  iputils-ping  net-tools \
    libxext6 libxrender1 libxtst6 libxi6 libfreetype6 fonts-dejavu \
    libgl1-mesa-glx libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*




# ------------------------------------------------------
# tracked vehicles Install dependencies for Clothoids
# ------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    ruby \
    ruby-dev \
    build-essential \
    ninja-build \
    python3-dev

# Install Ruby gems needed by rake
RUN gem install colorize && gem install rubyzip -v 2.3.2

# ------------------------------------------------------
# Clone the Clothoids repo and build it
# ------------------------------------------------------
WORKDIR /root

# Clone the repository
RUN git clone https://github.com/mfocchi/clothoids.git

WORKDIR /root/clothoids

# Initialize submodules
RUN git submodule update --init --recursive

# Compile using rake
RUN rake

# Install Python bindings (editable mode)
RUN pip3 install .


# -----------------------------
# tracked vehicles updates
# -----------------------------
RUN pip3 install --no-cache-dir  numpy==1.23.1 pandas==1.4.3 meshcat catboost glob2  scipy==1.10.1  
# -----------------------------
# install specific version of open3d not available by default with python3.8, need to upgrade pip to version >=23 
# -----------------------------
# file downloaded from https://pypi.org/project/open3d/0.18.0/#files
RUN python3 -m pip install --upgrade pip setuptools wheel

# ----------------------------------------
# Download the specific Open3D 0.18 wheel
# ----------------------------------------
RUN curl -fsSL \
    "https://files.pythonhosted.org/packages/ad/7c/a9369667ceed78c23da4503663825d7372c0b911829ee1e31628176c5936/open3d-0.18.0-cp38-cp38-manylinux_2_27_x86_64.whl" \
    -o /tmp/open3d-0.18.0-cp38-cp38-manylinux_2_27_x86_64.whl

RUN pip3 install --no-deps /tmp/open3d-0.18.0-cp38-cp38-manylinux_2_27_x86_64.whl

# Install only the minimal required deps
RUN pip3 install numpy addict tqdm pyquaternion Pillow dash scikit-learn



RUN apt-get update && apt-get install -y ros-noetic-velodyne-description   psmisc


# -----------------------------
# Patch roslaunch to reduce shutdown timeout
# -----------------------------
RUN python3 - <<'PY'
import re, pathlib
paths = [
    pathlib.Path("/opt/ros/noetic/lib/python3/dist-packages/roslaunch/nodeprocess.py"),
    pathlib.Path("/usr/lib/python3/dist-packages/roslaunch/nodeprocess.py"),
]
for p in paths:
    if p.exists():
        s = p.read_text()
        s = re.sub(r'(_TIMEOUT_SIGINT\s*=\s*)[0-9.]+', r'\g<1>0.1', s)
        s = re.sub(r'(_TIMEOUT_SIGTERM\s*=\s*)[0-9.]+', r'\g<1>0.1', s)
        p.write_text(s)
        print("Patched:", p)
        break
PY

# Source ROS automatically  use single quote to have it literal and avoid expansion of empty $PYTHONPATH
RUN echo '#!/bin/bash' >> /etc/bash.bashrc && \
    echo 'source /opt/ros/noetic/setup.bash' >> /etc/bash.bashrc && \
    echo '[ -f /root/ros_ws/install/setup.bash ] && source /root/ros_ws/install/setup.bash' >> /etc/bash.bashrc && \
    echo 'export LOCOSIM_DIR=/root/ros_ws/src/tracked_robot_simulation' >> /etc/bash.bashrc && \
    echo 'export PATH=/opt/openrobots/bin:$PATH' >> /etc/bash.bashrc && \
    echo 'export DISABLE_ROS1_EOL_WARNINGS=1' >> /etc/bash.bashrc && \    
    echo 'export PYTHONPATH=/opt/openrobots/lib/python3.8/site-packages:$LOCOSIM_DIR/robot_control:$PYTHONPATH' >> /etc/bash.bashrc && \
    echo 'export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/opt/openrobots/share/' >> /etc/bash.bashrc && \
    echo 'alias pycharm='\''/root/pycharm/bin/pycharm.sh'\''' >> /etc/bash.bashrc
RUN chmod +x /etc/bash.bashrc

# your container now always runs Bash as a login shell, /etc/bash.bashrc will be sourced automatically.
ENTRYPOINT []
# Start bash by default (interactive login shell)
CMD ["/bin/bash", "--login", "-i"]

