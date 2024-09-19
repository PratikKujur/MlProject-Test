import streamlit as st
from MlPipeline.DataIngestion import DataIngestion
from MlPipeline.DataTransformation import DataTransformation
from MlPipeline.ModelTraining import ModelTraining
from MlPipeline.ModelEvaluation import ModelEvaluation


def main():
    st.title("Model Comparison Pipeline with DagsHub (MLflow)")
    
    # Step 1: Data Ingestion
    st.header("1. Data Ingestion")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    ingestion = DataIngestion(url)
    dataframe = ingestion.load_data()
    st.write("Dataset Sample:", dataframe.head())

    # Step 2: Data Transformation
    st.header("2. Data Transformation")
    transformation = DataTransformation(dataframe)
    X, Y = transformation.transform()
    st.write("Features shape:", X.shape)
    st.write("Target shape:", Y.shape)

    # Step 3: Model Training
    st.header("3. Model Training and Evaluation")
    training = ModelTraining(X, Y)
    results, names = training.train_models()

    # Step 4: Model Evaluation
    st.header("4. Model Evaluation")
    evaluation = ModelEvaluation(results, names)
    evaluation.evaluate()


if __name__ == '__main__':
    main()








