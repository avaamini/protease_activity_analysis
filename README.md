# Protease Activity Analysis
Collection of tools for analyzing protease activity data in Python.

## Setup
1. Install the following dependencies: [conda 3.x](https://docs.conda.io/en/latest/miniconda.html)
2. [Generate Github SSH keys](https://help.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). If you already have SSH keys you can first [check to make sure](https://help.github.com/en/github/authenticating-to-github/checking-for-existing-ssh-keys).
3. [Add your SSH key to your Github account](https://help.github.com/en/github/authenticating-to-github/checking-for-existing-ssh-keys).
4. Download this repository: 
```
git clone git@github.com:apsoleimany/protease_activity_analysis.git
```


## Installation
To use the protease activity toolbox, first create the environment, and install the package.
```
cd protease_activity_analysis
conda env create -f environment.yml
conda activate protease
pip install -e .
```

## Usage
To enter the protease environment:
```
conda activate protease
```
To deactivate after you're done:
```
conda deactivate
```
