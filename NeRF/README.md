# Environment setup

## Create env
`conda env create -f env.yml`

## Build NeRF
```
cd instant-ngp
cmake . -B build
cmake --build build --config RelWithDebInfo -j 16
```

## Build COLMAP
```
cd colmap
mkdir ./build
cd ./build
cmake ../
make -j
make install
```

## Build ceres-solver
```
cd ceres-solver
mkdir ./build
cd ./build
cmake ../ -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
make -j
make install
```

# Run NeRF
```
cd instant-ngp
python scripts/colmap2nerf.py --video_in ../input_video/ph_sample.mp4 --run_colmap --video_fps 10 --out ../sample_video/transforms.json
./build/testbed --scene ../sample_video/
```
