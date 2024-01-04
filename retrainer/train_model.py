import os

def train_model(identifier):
    print(f"Model trained successfully with identifier: {identifier}")
    model_filename = f"model-{identifier}.pkl"

if __name__ == "__main__":
    identifier = os.getenv("IDENTIFIER", "default-identifier")
    train_model(identifier)
