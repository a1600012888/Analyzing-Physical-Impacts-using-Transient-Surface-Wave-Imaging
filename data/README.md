# Data Informations


We release the measured tilts in both x and y directions for a couple of experiments (experiments with both isotropic and anisotropic materials).  The measured tilts in saved either in `processed_final.npz` or `measurements_data.hdf5` in each folder. For visualization purposes, extra images or videos from a side-RGB camera might also be provided in each folder. We also provide the impact location and camera/material calibration results. 

To load, visualize and work with the data, please follow `../LocalizationMain.ipynb` and `../PingpngDemo.ipynb`. 

## Meaning of each data folder: 


| directory name                  | description |
|---------------------------------|----------|
| isotropic_plasticboard_20cm     | measurements from several knocks on an isotropic white plasticboard. The average distance from knocking locations to measurement points is around 20 cm.          |
| isotropic_plasticboard_30cm     |  measurements from several knocks on an isotropic white plasticboard. The average distance from knocking locations to measurement points is around 30 cm.         |
| isotropic_medium-density-fiberboard |  measurements from several knocks on an isotropic medium-density board.   |
| isotropic_glass                 |   measurements from several knocks on an isotropic medium glass.        |
| isotropic_particleboard         |  measurements from several knocks on an isotropic particleboard.         |
| birchlywood_anisotropy0.9       |   measurements from several knocks on an anisotropic (orthotropic) birch plywood, the anistotropy parameter is around 0.9.        |
| porcelain_anisotropy_3.0        |  measurements from several knocks on an anisotropic porcelain, the anistotropy parameter is around 3.0.         |
| soft-pvcboard_anisotropy0.6     |  measurements from several knocks on an anisotropic soft PVC panels, the anistotropy parameter is around 0.6             |
| heavyplywood_anisotropy1.5      |  measurements from several knocks on another anisotropic plywood, the anistotropy parameter is around 1.5       |
| --------------------------------|------------|
| pingpong_demo                   |   contains videos and measurements for the non-line-of-sight ping-pong demos       |
