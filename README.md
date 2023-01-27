## Intent Classification for virtual assistants
This is a project made for the Natural Language processing and Machine Learning
seminar.

This project contains 2 pytorch models:
- `model` - A simple model using pre-trained
    paraphrase-albert-small-v2 from HuggingFace without any
    fine-tuning
- `model2` - Model based on paraphrase-albert-small-v2 fine tuned with the 
    a part of the [Out-of-Scope Intent Classification Dataset](https://www.kaggle.com/datasets/stefanlarson/outofscope-intent-classification-dataset)

Additionally there's also a simple interactive mode to try out the
model.

### Requirements
- python >=3.10
- pytorch
- pandas
- scikit-learn
- transformers
- numpy

### Run interactive mode
* Download repository
    ```sh
    git clone https://github.com/vabenil/albert_intent_analysis
    ```
* Move to the root directory of the project
    ```sh
    cd albert_intent_analysis
    ```
* Run
    ```sh
    python -m intent_analysis.interactive
    ```

### Installation
Requires pip installed.

- Download repository
    ```sh
    git clone https://github.com/vabenil/albert_intent_analysis
    ```
- Move to the root directory of the project
    ```sh
    cd albert_intent_analysis
    ```
- Install package
    ```sh
    pip install .
    ```

### Usage
Example usage of model
```python
from intent_analysis.model import Net, run_model
model = Net()

# token representing sentence as (1, 768) vector
Y = model(["Hello!"])
labels = run_model(model, ["Hello!"])
```

Example usage of model2
```python
from intent_analysis.model2 import Net, run_model

model = Net.pretrained()
# one-hot-encoded vector representing label
Y = model(["Hello!"])
# List of labels. In this case ["GREETING"]
labels = run_model(model, ["Hello!"])
```

### Dataset
Training data is slightly modified version of 
    a part of the [Out-of-Scope Intent Classification Dataset](https://www.kaggle.com/datasets/stefanlarson/outofscope-intent-classification-dataset)
- train data (found in `datasets/train_data.json`)
- test data (found in `datasets/test_data.json`)
