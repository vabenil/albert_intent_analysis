from .model2 import Net, run_model
import torch


if __name__ == '__main__':
    model = Net.pretrained()

    while True:
        user_input = input("You: ")
        if user_input.upper() == "/QUIT":
            break

        label = run_model(model, [user_input])[0]
        print(f"LABEL: {label}")

