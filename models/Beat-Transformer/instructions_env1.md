# Install pyenv if you don't have it already (one-time setup):
brew install pyenv

# Install a compatible Python version using pyenv (e.g., 3.9.17)
pyenv install 3.9.17

# Set the local Python version for the project (this creates a .python-version file in the current directory)
pyenv local 3.9.17

# Verify the Python version is correct (should show Python 3.9.17)
python --version

# Create and activate a virtual environment using the local Python (env1)
python -m venv env1
source env1/bin/activate

# Now install Spleeter 2.3.2 and librosa; with Python 3.9, the version constraint should be met.
pip install spleeter==2.3.2 librosa
