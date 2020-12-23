pip install pyyaml
pip install numpy 
pip install scipy
pip install matplotlib
pip install cython
pip install opencv-python
pip install pycocotools
pip install pytest
pip install pybind11

cd layers/anchor_gen_cuda
python3 setup.py install
cd ../nms_cuda
python3 setup.py install
cd ../sigmoid_focal_loss_cuda
python3 setup.py install
cd ../assign_cuda
python3 setup.py install
cd ../roi_align_corners_cuda
python3 setup.py install
cd ../deform_im2col_cuda
python3 setup.py install
cd ../DCNv2
python setup.py build develop

cd ../../dataset/dbext
python3 setup.py install
