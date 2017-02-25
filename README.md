Python Product Category Classification Model
============================================

## Project' structure ##

docs/
    Simple_Product_Category_Classification.ipynb    Ipython notebook with samples and tutorials

prod_classify/
    core.py                                         Implementation of product category classification model

resources/
    Products.csv                                    Train dataset

tests/
    test_prod_classify.py                           Tests of product category classification model

main.py                 REST service

LICENSE                 License file

README.md               this file with short description of project

requirements.txt        Requirements of libraries and packages

setup.py                Setup file for package installation

## Environment setup ##

Install virtualenv

```bash
sudo apt install python-pip
pip install virtualenvwrapper
```

Add next lines to ~/.bashrc (~/.profile)

```bash
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/Devel

# load virtualenvwrapper for python (after custom PATHs)
venvwrap="virtualenvwrapper.sh"
/usr/bin/which $venvwrap
if [ $? -eq 0 ]; then
    venvwrap=`/usr/bin/which $venvwrap`
    source $venvwrap
fi
```

Run script

```bash
. ~/.local/bin/virtualenvwrapper.sh
```

Create virtual environment

```bash
mkvirtualenv -p python3.5 prod_classify
```

## Install dependencies ##

```bash
pip install -r requirements.txt
```

## install the package locally (for use on our system) ##

```bash
pip install -e .
```

## Run tests ##

```bash
pytest tests
```

## Run demo app ##

```bash
python main.py
```