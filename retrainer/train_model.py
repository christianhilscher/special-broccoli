import os
import time
import random
import string

def train_model(identifier):
    print(f"Model trained successfully with identifier: {identifier}")
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    model_filename = f"/models/model-{identifier}.pkl"
    with open(model_filename, "w") as file:
        file.write("dummy model data" + random_string)

if __name__ == "__main__":
    identifier = os.getenv("IDENTIFIER", "default-identifier")
    train_model(identifier)
    time.sleep(5)