import os

def train_model(identifier):
    print(f"Model trained successfully with identifier: {identifier}")
    model_filename = f"/home/ec2-user/models/model-{identifier}.pkl"
    with open(model_filename, "w") as file:
        file.write("dummy model data")

if __name__ == "__main__":
    identifier = os.getenv("IDENTIFIER", "default-identifier")
    train_model(identifier)
