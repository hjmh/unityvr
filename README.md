# unityvr
Tools for analyzing data collected with a custom unity-based VR for insects.

### Organization:
The unityvr package contains the following submodules:
* **preproc**
* **viz**

In addition the **scrips** folder contains notebooks that illustrate how to use functions in this module based on an example file in **sample** (sampleLog.json).


### Install:
I recommend using poetry to setup a custom conda environment. A helpful introduction can be found [here](https://ealizadeh.com/blog/guide-to-python-env-pkg-dependency-using-conda-poetry).

0. Clone repo, navigate into folder
1. If you don't already have poetry, [install poetry](https://python-poetry.org/docs/#installation). You may need to close command window and open a new one.
2. Create conda environment:  
 `conda create --name unityvr python=3.8`
4. Activate environment:  
 `conda activate unityvr`
6. Make sure you are in the top folder of the cloned repo, then install dependencies:  
 `poetry install`
8. Setup the new environment as an ipython kernel:  
    `conda install -c anaconda ipykernel`  
    then  
    `python -m ipykernel install --user --name=unityvr`
    
Now you should be able to run the example notebooks in the **scripts** folder without problems. 
