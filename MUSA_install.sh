MUSA_ARCH=22 FORCE_MUSA=1 MMCV_WITH_OPS=1 pip install -e . -v
new_path="/home/mmcv/build/MMCV/lib"

if ! grep -q "export LD_LIBRARY_PATH=$new_path:\$LD_LIBRARY_PATH" ~/.bashrc; then
	echo "export LD_LIBRARY_PATH=$new_path:\$LD_LIBRARY_PATH" >> ~/.bashrc
fi

source ~/.bashrc
echo "mmcv lib is /home/mmcv/build/MMCV/lib, please do not delete it!"
