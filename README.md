# NCKRIG

Software to reconstruct missing data in climate datasets using Kriging

## Dependencies
- python>=3.11.0
- pykrige>=1.7.0
- tqdm>=4.64.1
- numpy>=1.23.5
- xarray>=2022.11.0
- netcdf4>=1.6.2
- matplotlib>=3.7.1

An Anaconda environment with all the required dependencies can be created using `environment.yml`:
```bash
conda env create -f environment.yml
```
To activate the environment, use:
```bash
conda activate nckrig
```

## Installation

`nckrig` can be installed using `pip` in the current directory:
```bash
pip install .
```

## Usage

```
nckrig <netcdf-filename> <varname>
```

## License

`nckrig` is licensed under the terms of the BSD 3-Clause license.
