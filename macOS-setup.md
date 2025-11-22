# pyenv

```
brew install pyenv
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> /Users/ching-yichen/.zprofile
source /Users/ching-yichen/.zprofile

nano ~/.zshrc
export PATH="$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"


pyenv install 3.10.12  # or whichever patch version
pyenv global 3.10.12
cd /path/to/your/project
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cd research_project

python index.py
```


# Manage py versions
```
pyenv install 3.10.12

pyenv global 3.10.12

or pyenv local 3.10.12

//Recreate
rm -rf .venv

python -m venv .venv

source .venv/bin/activate

# pip install -r requirements.txt

pip install tensorflow-macos==2.10

pip install matplotlib numpy opencv-python==4.9.0.80 pyautogui PyYAML torch torchvision==0.16.2 Cython PyGetWindow

cd research_project

python index.py
```