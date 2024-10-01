# SyntheOcc

> SyntheOcc: Synthesize Geometric-Controlled Street View Images through 3D Semantic MPIs <br>
> [Leheng Li](https://len-li.github.io), Weichao Qiu, Yingjie Cai, Xu Yan, Qing Lian, Bingbing Liu, Ying-Cong Chen

SyntheOcc is a project focused on synthesizing image data under geometry control (condition occupancy). This repository provides tools and scripts to process, train, and generate synthetic image data in the nuScenes dataset.
#### [Project Page](https://len-li.github.io/syntheocc-web) | [Paper](https://arxiv.org/) | [Video](https://bilibili.com)


## Table of Contents

- [SyntheOcc](#syntheocc)
      - [Project Page | Paper | Video](#project-page--paper--video)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Prepare Dataset](#prepare-dataset)
  - [Train](#train)
  - [Visualize](#visualize)




## Installation

To get started with SyntheOcc, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/EnVision-Research/SyntheOcc.git
   cd SyntheOcc
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Prepare Dataset

To use SyntheOcc, follow the steps below:

1. **Download the NuScenes dataset:**
   - Register and download the dataset from the [NuScenes website](https://www.nuscenes.org/nuscenes).
   - Extract the dataset to a suitable location: `python scripts/extract_nuscenes_data.py`

2. **Configure the dataset path:**
   - Update the `config.yaml` file with the path to your NuScenes dataset.

3. **Run the occupancy synthesis script:**
   ```bash
   python scripts/synthesize_occupancy.py
   ```


## Train 

   ```bash
   bash train.sh
   ```

## Visualize 

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


