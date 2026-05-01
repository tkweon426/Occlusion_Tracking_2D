# Occlusion_Tracking_2D
A simulation setup for Occlusion aware tracking on a 2D scene for optimal control and estimation 

## Setting up the Repository

Clone the repository first:
```
git clone https://github.com/tkweon426/Occlusion_Tracking_2D/
```

Setup the conda environment file
```
conda env create -f environment.yml
```

Activate the conda environment 
```
conda activate tracking_sim
```

## Running the Code
```
python3 main.py
```

Arguments to modify the run 
```
--record
# records the run as a 60fps video

--log
# logs the simulation run information, including compute times for controller

--leave_trajectory
# leaves trajectory as low opacity lines 
```