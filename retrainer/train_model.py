import typer

def train_model(identifier: str):
    print(f"Model trained successfully with identifier: {identifier}")
    model_filename = f"model-{identifier}.pkl"

if __name__ == "__main__":
    typer.run(train_model)
