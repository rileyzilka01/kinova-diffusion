import libtmux
import time
from os import path
import sys

if __name__ == "__main__":
	server = libtmux.Server(
		config_file=path.expandvars("/home/user/kinova-diffusion/scripts/.tmux.conf")
	)
	model = sys.argv[1]
	if model not in ["hitl_d", "hitl_hgd"]:
		print("Model does not exist, must be <hitl_d, hitl_hgd>")
		sys.exit(1)

	local = sys.argv[2]
	if local not in ["0", "1"]:
		print("local must be 0 or 1")
		exit(1)
	if server.has_session("sim"):
		exit()
	else:
		session = server.new_session("sim", start_directory="/home/user/kinova-diffusion", attach=False)
		
	# terminals for the simulation to start
	terminals = {
		"kortex_bringup": "roslaunch kortex_bringup kortex_bringup.launch", # launch kortex - note that this starts a roscore
		"main": "sleep 1 && rosrun kortex_bringup inference_main.py", 
		"joy": "rosrun joy joy_node", 
		"segment": "rosrun kortex_bringup segment.py" if model == "hitl_hgd" else "echo 'No segmentation for hitl-d'", 
		"realsense_back": "sleep 5 && roslaunch realsense2_camera rs_camera.launch camera:=cam filters:=pointcloud depth_width:=640 depth_height:=480 depth_fps:=30 color_width:=640 color_height:=480 color_fps:=30 align_depth:=true decimation_filter:=true spatial_filter:=true temporal_filter:=true hole_filling_filter:=true",
		"inference": f"sleep 1 && rosrun kortex_bringup inference.py {model} {local}",
		"rviz": "sleep 5 &&rviz -d /home/user/kinova-diffusion/scripts/hitl_hgd.rviz" if model == "hitl_hgd" else "sleep 5 &&rviz -d /home/user/kinova-diffusion/scripts/hitl_d.rviz",
	}

	for name, cmd in terminals.items():
		window = session.new_window(name, attach=False)
		window.select_layout(layout="tiled")
		pane = window.panes[0]
		time.sleep(0.1)
		pane.send_keys(cmd, suppress_history=True)