pipeline {
    agent any  

    environment {
        VENV_DIR = 'venv'  
        TRAIN_PATH = "data/train.csv"
        TEST_PATH = "data/test.csv"
        MODEL_PATH = "models/xgboost_model.pkl"

    }

    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'main', url:'https://github.com/Leila222/MLops.git' 
            }
        }

        stage('Set up Environment') {
            steps {
                sh 'python3 -m venv ${VENV_DIR}'
                sh '. ${VENV_DIR}/bin/activate && pip install -r requirements.txt'
            }
        }

        stage('Prepare Data') {
            steps {
                sh 'source ${VENV_DIR}/bin/activate && python main.py --prepare --train_path ${TRAIN_PATH} --test_path ${TEST_PATH}'
            }
        }

        stage('Train Model') {
            steps {
                sh 'source ${VENV_DIR}/bin/activate && python main.py --train --train_path ${TRAIN_PATH} --test_path ${TEST_PATH} --model_path ${MODEL_PATH}'
            }
        }

        stage('Evaluate Model') {
            steps {
                sh 'source ${VENV_DIR}/bin/activate && python main.py --evaluate --train_path ${TRAIN_PATH} --test_path ${TEST_PATH} --model_path ${MODEL_PATH}'
            }
        }

        stage('Retrain Model') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Retrain Model' }
            }
            steps {
                sh '''
                . ${VENV_DIR}/bin/activate 
                if [ -f ${MODEL_PATH} ]; then
                    echo "Existing model found, retraining..."
                    python main.py --train --train_path ${TRAIN_PATH} --test_path ${TEST_PATH} --model_path ${MODEL_PATH}
                else
                    echo "No model found, training a new model..."
                    python main.py --train --train_path ${TRAIN_PATH} --test_path ${TEST_PATH} --model_path ${MODEL_PATH}
                fi
                '''
            }
        }
        
        stage('Deploy API') {
            steps {
                sh 'source ${VENV_DIR}/bin/activate && python app.py' 
            }
        }
    }
}



