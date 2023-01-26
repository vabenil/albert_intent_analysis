from model import Net
from simple import run_model


if __name__ == '__main__':
    model = Net()
    while True:
        user_input = input("You: ")
        if user_input.upper() == "/QUIT":
            break

        label = run_model(model, [user_input])[0]
        print(label)

