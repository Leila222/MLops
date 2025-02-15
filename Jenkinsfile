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
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Deploy API' }
            }
            steps {
                sh '. ${VENV_DIR}/bin/activate && python app.py &'
            }
        }
    }
}

