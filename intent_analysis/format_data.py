import json

LABELS  = [
    'GREETING', 'DATE', 'WEATHER', 'REPEAT', 'YES',
    'CANCEL', 'FLIP_COIN', 'TRANSLATE', 'TIMER', 'DEFINITION',
    'MEANING_OF_LIFE', 'WHAT_CAN_I_ASK_YOU', 'ROLL_DICE', 'MAKE_CALL', 'ALARM'
]

def format_file(path: str) -> dict[str, list[str]]:
    json_data = None
    with open(path, 'r') as f:
        json_data = json.load(f)

    formatted_data = {}
    for sentence, label in json_data:
        label = label.upper()
        if label in LABELS:
            if label in formatted_data:
                formatted_data[label].append(sentence)
            else:
                formatted_data[label] = [sentence]

    return formatted_data

if __name__ == '__main__':
    formatted_data = format_file('datasets/is_train.json')
    with open('datasets/train_data.json', 'w') as f:
        json.dump(formatted_data, f, sort_keys=True, indent=4)
