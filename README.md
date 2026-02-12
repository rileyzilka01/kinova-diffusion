# How to run kinova-diffusion
```bash
docker compose build --no-cache
```

Start the docker service which provides access to the ros nodes
```bash
docker compose up -d
```

To stop the detached docker process when in directory
```bash
docker compose down kinova_diffusion
```

Enter interactive terminal
```bash
docker exec -it png_vision bash
```

# Starting Rvis
In container terminal
```bash
rviz
```
If error go into system terminal and try again
```bash
xhost +
```

# Starting the project nodes

Adding container bash to alias, type kd_bash to get into docker container terminal
```bash
alias kd_bash='docker exec -it kinova_diffusion bash' >> ~/.bashrc
```

- Turn Kinova on by holding power button til green light
- Ensure Kinova is plugged in and shows connection in wired connections
	- Can verify connection by going to the ip in google and using the interface
- Ensure Camera is plugged into USB port
- Ensure joystick is plugged in

# Example runs
### Recording data
```bash
python3 scripts/record.py && tmux attach
```

### Inference
```bash
python3 scripts/inference.py && tmux attach
```

### TMUX Info
To end the tmux
```
CTRL + b then d
```
```bash
tmux kill-session
```

If the launch is saying "No realsense devices found" follow the steps
1. attempt to replug all cords
2. Can check if its working by doing
```bash
sudo dmesg -w
```
Can also check plugged in status with
```bash
lsusb
```
or
```bash
lsusb -t
```
3. Check 
```bash
ls /dev
```
on both the machine and in the docker container
If the numbers are different just recompose the docker container

4. just restart the pc to fix usb hub

# If the realsense camera break, update noetic
```bash
sudo apt install librealsense2=2.55.1-0~realsense.12473 librealsense2-gl=2.55.1-0~realsense.12473 librealsense2-utils=2.55.1-0~realsense.12473 librealsense2-dev=2.55.1-0~realsense.12473
```

# If camera breaks with error about realsense2_camera/RealSenseNodeFactory
```bash
sudo apt install ros-noetic-realsense2-camera
```

# Libraries
This makes use of the 
kinova ros_kortex package at https://github.com/Kinovarobotics/ros_kortex 
kortex_bringup package from https://github.com/cjiang2/kortex_bringup
