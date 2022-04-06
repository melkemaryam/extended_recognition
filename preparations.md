# Preparations

### 1. Install Virtualenv:

`$ pip3 install virtualenv virtualenvwrapper`

### 2. Edit `~/.bashrc` profile

`$ vim ~/.bashrc`
	
```	# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.local/bin/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=$HOME/Library/Python/3.8/bin/virtualenv
source $HOME/Library/Python/3.8/bin/virtualenvwrapper.sh
```
### 3. Source venv

`$ source ~/.bashrc`

### New terminal commands:

* Create an environment with `mkvirtualenv`
* Activate an environment (or switch to a different one) with `workon`
* Deactivate an environment with `deactivate`
* Remove an environment with `rmvirtualenv`

more infos [in the docs](https://virtualenvwrapper.readthedocs.io/en/latest/)

### 4. Create new venv:

`$ mkvirtualenv traffic_signs -p python3`

### 5. Install packages:

```
$ workon traffic_signs
$ pip install opencv-contrib-python
$ pip install numpy
$ pip install scikit-learn
$ pip install scikit-image
$ pip install imutils
$ pip install matplotlib
$ pip install tensorflow # or tensorflow-gpu
```

## Note

A virtual environment is not necessary in order to run the framework, however, it is recommended in order to protect the data and devices.

If working without a virtual environment, please make sure that all necessary libraries are installed. If not, install them like this:

```
$ pip3 install opencv-contrib-python
$ pip3 install numpy
$ pip3 install scikit-learn
$ pip3 install scikit-image
$ pip3 install imutils
$ pip3 install matplotlib
$ pip3 install tensorflow # or tensorflow-gpu
```