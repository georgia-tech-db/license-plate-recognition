python3 -m venv eva-application-venv
source eva-application/bin/activate-venv

pip install -r requirements.txt

# Download model
wget "https://www.dropbox.com/s/x677jwtae0elm6h/model.pth?dl=0"
mv model.pth\?dl=0 model.pth
