###########################################################
###   
###   ███████╗██╗░░░██╗░█████╗░
###   ██╔════╝██║░░░██║██╔══██╗
###   █████╗░░╚██╗░██╔╝███████║
###   ██╔══╝░░░╚████╔╝░██╔══██║
###   ███████╗░░╚██╔╝░░██║░░██║
###   ╚══════╝░░░╚═╝░░░╚═╝░░╚═╝
###   
###########################################################

# Setup virtual environment
python3 -m venv eva-application-venv
source eva-application/bin/activate-venv

# Install EVA application dependencies
pip install -r requirements.txt

# Download car plate detection model (based on image segmentation)
wget "https://www.dropbox.com/s/x677jwtae0elm6h/model.pth?dl=0"
mv model.pth\?dl=0 model.pth

# Refer the custome user-defined function (UDF)
cat car_plate_detector.py

# Convert Jupyter notebook to README markdown
jupyter nbconvert --execute --to markdown README.ipynb