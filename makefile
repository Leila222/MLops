VENV = venv
PYTHON = $(VENV)/bin/python
TRAIN_PATH=data/train.csv
TEST_PATH=data/test.csv
MODEL_PATH=models/xgboost_model.pkl
FLASK_APP = app.py

all: install prepare train evaluate deploy

install: 
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

check_code:
	$(PYTHON) -m black .  
	$(PYTHON) -m flake8 .  
	$(PYTHON) -m bandit -r . 

test:
	$(PYTHON) -m pytest tests/

prepare:
	$(PYTHON) main.py --prepare --train_path $(TRAIN_PATH) --test_path $(TEST_PATH)

train:
	$(PYTHON) main.py --train --train_path $(TRAIN_PATH) --test_path $(TEST_PATH) --model_path $(MODEL_PATH)

evaluate:
	$(PYTHON) main.py --evaluate --train_path $(TRAIN_PATH) --test_path $(TEST_PATH) --model_path $(MODEL_PATH)
	
deploy:
	$(PYTHON) app.py
		
clean:
	rm -rf $(VENV)
	rm -rf models/

help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  prepare   - Prepare the dataset"
	@echo "  train     - Train the model and save it"
	@echo "  evaluate  - Evaluate the model"
	@echo "  explain   - Explain churn for a customer (set EXPLAIN_INDEX)"
	@echo "  clean     - Remove generated model files"
