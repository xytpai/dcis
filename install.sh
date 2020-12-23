pip install pyyaml
pip install numpy 
pip install scipy
pip install matplotlib
pip install cython
pip install opencv-python
pip install pycocotools
pip install pytest
pip install pybind11

cd layers/dcis_ext
python3 setup.py install
cd ../DCNv2
python setup.py build develop
