include .env
export

SHELL := /bin/bash
SRC = */*.py
DATA = data/processed/train.csv data/test/dftest.csv
PREP_MODEL = models/prep.model
MLFLOW_RUN = mlflow run

all: train serve
train: train_autogluon
tmp/run_env.sh: $(SRC) $(DATA)
	$(MAKE) train

train_h2o: $(SRC) $(DATA) $(PREP_MODEL)
	$(MLFLOW_RUN) h2o_mlflow -e train_h2o

train_autogluon: $(SRC) $(DATA) $(PREP_MODEL)
	$(MLFLOW_RUN) autogluon_mlflow -e train_autogluon

preproc $(DATA) $(PREP_MODEL): common/preprocess.py
	$(MLFLOW_RUN) common -e preprocess

devenv: conda*.yml
	for e in */conda.yml; do conda env update -f $$e; done
	conda env update -f conda-dev.yml
	python --version
	gcc --version
	java -version

cleancode:
	bash scripts/clean_code.sh

clean:
	rm -Rf tmp
