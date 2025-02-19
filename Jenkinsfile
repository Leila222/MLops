pipeline {
    agent any  

    environment {
        VENV_DIR = 'venv'  
        TRAIN_PATH = "data/train.csv"
        TEST_PATH = "data/test.csv"
        MODEL_PATH = "models/xgboost_model.pkl"
    }

    parameters {
        string(name: 'RUN_STAGE', defaultValue: 'ALL', description: 'Enter stage name to run a single stage or ALL to run everything')
    }

    stages {
        stage('Checkout Code') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Checkout Code' }
            }
            steps {
                git branch: 'main', url:'https://github.com/Leila222/MLops.git' 
            }
        }

        stage('Set up Environment') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Set up Environment' }
            }
            steps {
                sh 'python3 -m venv ${VENV_DIR}'
                sh '. ${VENV_DIR}/bin/activate && pip install -r requirements.txt'
            }
        }
        
        stage('Deploy API') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Run MLflow' }
            }
            steps {
                sh '. ${VENV_DIR}/bin/activate && mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5001 &'
            }
        }

        stage('Prepare Data') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Prepare Data' }
            }
            steps {
                sh '. ${VENV_DIR}/bin/activate && python main.py --prepare --train_path ${TRAIN_PATH} --test_path ${TEST_PATH}'
            }
        }

        stage('Train Model') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Train Model' }
            }
            steps {
                sh '. ${VENV_DIR}/bin/activate && python main.py --train --train_path ${TRAIN_PATH} --test_path ${TEST_PATH} --model_path ${MODEL_PATH}'
            }
        }

        stage('Evaluate Model') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Evaluate Model' }
            }
            steps {
                sh '. ${VENV_DIR}/bin/activate && python main.py --evaluate --train_path ${TRAIN_PATH} --test_path ${TEST_PATH} --model_path ${MODEL_PATH}'
            }
        }

        stage('Deploy API') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Deploy API' }
            }
            steps {
                sh '. ${VENV_DIR}/bin/activate && python app.py'
            }
        }
    }
}
