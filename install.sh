# Create the virtual environment
python3 -m venv --prompt SATINET venv
source venv/bin/activate 

# Upgrade the pip and setuptools
pip install --upgrade pip
python3 -m pip install --upgrade setuptools

# Update the submodules
git submodule init
git submodule update

# Install the required packages
pip install -r requirements.txt


#pip install progressist==0.1.0
#pip install tensorflow==1.14.0
#pip install numpy==1.16.4
#pip install opencv-contrib-python==3.4.2.16
#pip install opencv-python==3.4.2.16
#pip install tqdm==4.50.0
