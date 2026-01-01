

CPP KODUNU DERLERKEN
cd ~/Desktop/face_samples 
önceden build ettiysen sil:
rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)
./faceid

