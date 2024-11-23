sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv
python3.12 -m venv .dev

. .dev/bin/activate

pip install -r requirements.txt