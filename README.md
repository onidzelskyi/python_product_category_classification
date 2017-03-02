Python Product Category Classification Model
============================================

## Project' structure ##

docs/
    Simple_Product_Category_Classification.ipynb    Ipython notebook with samples and tutorials

prod_classify/
    __init__.py                                     Package file
    core.py                                         Implementation of product category classification model
    endpoints.py                                    Implementation of RESTful application endpoints.

resources/
    bad_csv_content.csv                             Data for tests
    bad_json_content.json                           Data for tests
    model.pickle                                    Data for tests
    Products.csv                                    Train dataset
    test_set.csv                                    Data for tests
    train_set.csv                                   Data for tests
    train_set.json                                  Data for tests

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

## Train model from CSV file ##

```bash
curl -F file=@resources/train_set.csv localhost:5000/fit
```

## Train model from JSON data ##

```bash
curl -X POST -H "Content-Type: application/json" localhost:5000/fit -d @resources/train_set.json
```

## Predict product' category ##

```bash
curl -X POST -H "Content-Type: application/json" localhost:5000/predict -d '{"products": {"101": "best"}}'
```

## Get model statistics (accuracy) ##

```bash
curl -X POST localhost:5000/statistics -F file=@resources/test_set.csv
```

## Dump model to default file ##

```bash
curl -X GET localhost:5000/dump
```


## Dump model to given file ##

```bash
curl -X GET localhost:5000/dump?file=abc.txt
```

## Check PEP008 code style ##

```bash
flake8  --max-line-length=120 .
```