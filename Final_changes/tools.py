from transformers import BertTokenizer, BertModel
import torch
import pickle

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()
MODEL_NAME = "bert-base-uncased"

def text2embedding(sentence):
    if not sentence or not sentence.strip():
        return None

    inputs = tokenizer(sentence,return_tensors="pt",truncation=True,max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling over valid tokens only
    mask = inputs["attention_mask"].unsqueeze(-1)
    hidden = outputs.last_hidden_state

    embedding = ((hidden * mask).sum(dim=1)/ mask.sum(dim=1).clamp(min=1))

    return embedding


def create_user_embedding():
    user_profiles = []
    with open("C:/Users/Najari/Desktop/InformationDiffusion/Randomness/LAIDSim3--stance,/data/input/user_profile.txt", "r") as file:
        for line in file.readlines():
            profile = line.split(": ")[1]
            user_profiles.append(profile)

    user_embeddings = []
    for user in user_profiles:
        profile = remove_substrings(user, ['is a', 'enjoys', ' who ', '-year-old', 'He', 'She'])
        embedding = text2embedding(profile)
        user_embeddings.append(embedding)

    save_pickle(user_embeddings, "C:/Users/Najari/Desktop/InformationDiffusion/Randomness/LAIDSim3--stance,/data/input/user_embedding.pickle")


def remove_substrings(input_string, substrings_to_remove):
    result = input_string
    for substring in substrings_to_remove:
        result = result.replace(substring, "")
    return result


def save_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f'{filename} saved!')


def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        print(f'{filename} loaded!')
        return data


if __name__ == '__main__':
    # test
    create_user_embedding()
    # data = load_pickle('../data/input/user_embedding.pickle')
    # print(data)