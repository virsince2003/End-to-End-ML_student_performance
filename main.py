from src.components.data_ingestion import DataIngestion , DataIngestionConfig
from src.components.data_transform import DataTransform , DataTransform
from src.components.model_trainer import ModelTrain

def main():
    # Data ingestion
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()
    
    # Data transformation
    transformation = DataTransform()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(train_data_path, test_data_path)
    
    # Model training
    model_train = ModelTrain()
    r2_score = model_train.initate_model_training(train_arr, test_arr)
    
    print("R^2 Score:", r2_score)

if __name__ == "__main__":
    main()
