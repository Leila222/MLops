pipeline {
    agent any  

    environment {
        VENV_DIR = 'venv'  
        TRAIN_PATH = "data/train.csv"
        TEST_PATH = "data/test.csv"
        MODEL_PATH = "models/xgboost_model.pkl"
        RETRAINED_MODEL_PATH = "models/xgboost_retrained.pkl"
    }

    parameters {
        string(name: 'RUN_STAGE', defaultValue: 'ALL', description: 'Enter stage name to run a single stage or ALL to run everything')
        string(name: 'LEARNING_RATE', defaultValue: '0.1', description: 'Learning rate for retraining the model')
        string(name: 'MAX_DEPTH', defaultValue: '3', description: 'Max depth for the model')
        string(name: 'N_ESTIMATORS', defaultValue: '100', description: 'Number of estimators for the model')
        string(name: 'SUBSAMPLE', defaultValue: '1.0', description: 'Subsample ratio of the training data')
        string(name: 'COLSAMPLE_BYTREE', defaultValue: '1.0', description: 'Subsample ratio of features')
        string(name: 'GAMMA', defaultValue: '0', description: 'Minimum loss reduction required to make a further partition')
        string(name: 'MIN_CHILD_WEIGHT', defaultValue: '1', description: 'Minimum sum of instance weight (hessian) for a child')
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
        
        stage('Run MLflow') {
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

        stage('Retrain Model') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Retrain Model' }
            }
            steps {
                script {
                    
                    def learning_rate = params.LEARNING_RATE
                    def max_depth = params.MAX_DEPTH.toInteger()
                    def n_estimators = params.N_ESTIMATORS.toInteger()
                    def subsample = params.SUBSAMPLE.toDouble()
                    def colsample_bytree = params.COLSAMPLE_BYTREE.toDouble()
                    def gamma = params.GAMMA.toDouble()
                    def min_child_weight = params.MIN_CHILD_WEIGHT.toInteger()
                    def retrained_model_path = "models/xgboost_retrained.pkl"


                    sh """
                        . ${VENV_DIR}/bin/activate
                        echo "Retraining model with the following parameters:"
                        echo "learning_rate=${learning_rate}, max_depth=${max_depth}, n_estimators=${n_estimators}, subsample=${subsample}, colsample_bytree=${colsample_bytree}, gamma=${gamma}, min_child_weight=${min_child_weight}"

                       
                        python main.py --retrain --train_path ${TRAIN_PATH} --test_path ${TEST_PATH} --model_path ${retrained_model_path} \
                                       --learning_rate ${learning_rate} --max_depth ${max_depth} --n_estimators ${n_estimators} \
                                       --subsample ${subsample} --colsample_bytree ${colsample_bytree} --gamma ${gamma} \
                                       --min_child_weight ${min_child_weight}
                    """
                    
                    sh '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe Start-ScheduledTask -TaskName Jenkins_Notification'
                }
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

