echo "Updating system and installing prerequisites..."
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y software-properties-common wget git
echo "Installing Python 3.10..."
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-pip
echo "Verifying Python 3.10 installation..."
python3.10 --version
# 1. Install dependencies
sudo apt update
sudo apt install -y curl apt-transport-https ca-certificates gnupg

# 2. Add GitHub’s GPG key
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg

sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg

# 3. Add the gh CLI apt source
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] \
  https://cli.github.com/packages stable main" \
  | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null

# 4. Update and install
sudo apt update
sudo apt install -y gh

echo "Installing gdown..."
python3.10 -m pip install gdown
echo "Verifying gdown installation..."
python3.10 -m pip show gdown
echo "Cloning the IG-VLM-Modified repository..."
git clone https://github.com/DinhVitCuong/IG-VLM-Moddified /workspace/IG-VLM-Moddified
echo "Creating and activating virtual environment vqa_env..."
python3.10 -m venv /workspace/vqa_env
source /workspace/vqa_env/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install wheel sentencepiece opencv-python moviepy "numpy<2" transformers transformers-stream-generator pandas av protobuf accelerate

pip install flash-attn --no-build-isolation
