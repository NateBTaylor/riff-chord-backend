# Create and activate virtual environment for beat tracking (env_2)
python3 -m venv env2
source env2/bin/activate

# Install PyTorch (adjust version/extra-index-url if needed)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Install additional dependencies for madmom
pip install Cython mido

# Clone and build madmom from source (using only relevant commands)
git clone --recursive https://github.com/CPJKU/madmom.git && mv madmom tmp && mv tmp/* . && rm -rf tmp
python setup.py develop --user

# Clone the Beat Transformer repository (if not already cloned)
git clone --branch=main https://github.com/zhaojw1998/Beat-Transformer

# Move required model files to the project root (do not duplicate existing code)
mv ./Beat-Transformer/code/DilatedTransformer.py ./DilatedTransformer.py
mv ./Beat-Transformer/code/DilatedTransformerLayer.py ./DilatedTransformerLayer.py

# Run the beat tracking demo script
python /Users/nghiaphan/Desktop/Beat-Transformer/beat_tracking_demo.py
