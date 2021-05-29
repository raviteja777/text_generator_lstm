from process_data import DataProcessor
from model_params import set_model_params
from neural_network import ModelTrainer

import torch
import os
import random
from _datetime import datetime

DIR_PATH = "data/sonnets"
EPOCHS = 30
TRAIN_MODE = False  #trains model only if this value is true
SAVE_MODEL_PATH = "save_models"
VOCAB = DataProcessor.create_vocab()


def disp_msg(msg):
    print("=" * 5, msg, "=" * 5)


def invert_vocab():
    return dict({(v, k) for k, v in VOCAB.items()})


def process_data():
    disp_msg("begin file processing")
    processor = DataProcessor(DIR_PATH)
    disp_msg("dataset created")
    return processor


def generate_model(dataset, input_length):
    model_params = set_model_params(embedding_dim=input_length, hidden_dim=50, vocab_size=len(VOCAB), epochs=EPOCHS)
    disp_msg("training_model")
    trainer = ModelTrainer(model_params, dataset)
    disp_msg("Training completed")
    return trainer.get_model()


# generate text from model
# extra seed > 0 , specifies if to insert random values in between sequence
def gen_text(model, extra_seed=0):
    inv_vocab = invert_vocab()
    inp_len = model.word_embeddings.embedding_dim
    disp_msg("Generating sample text from model")
    disp_msg("===== ===== =====")
    para_len = random.randint(7, 20)
    for _ in range(0, para_len):
        line = []
        seed = random.randint(1, 26)
        line.append(seed)
        line += [0] * (inp_len - 1)
        seed_ins = False
        for i in range(0, inp_len):
            if extra_seed > 0 and i % (inp_len // extra_seed) == 5:
                seed_ins = True
            if seed_ins and line[i] == VOCAB[" "]:
                ind = random.randint(1, 26)
                seed_ins = False
            else:
                next_token = model(torch.tensor(line))
                ind = torch.argmax(next_token[i], 0)
                ind = ind.item()
                # print(ind)
            if i < inp_len - 1:
                line[i + 1] = ind
            else:
                line.append(ind)
        print(*[inv_vocab[i] for i in line if i > 0], sep="")


def save_model(model):
    timestamp = datetime.now().strftime("_%y%m%d_%H%M%S")
    model_filename = os.path.join(SAVE_MODEL_PATH, "model" + timestamp + ".ser")
    torch.save(model, model_filename)
    disp_msg("Model saved at path: " + model_filename)


if __name__ == '__main__':
    # keep TRAIN_MODE -- True to train and save model

    if TRAIN_MODE:
        processor = process_data()
        dataset = processor.create_dataset()
        input_length = processor.get_inp_len()
        print("Seq length ", input_length)
        model = generate_model(dataset, input_length)
        save_model(model)
        gen_text(model)
    else:
        # used for generating text from saved model
        # manually enter the save model path
        model_path = "save_models/model_sonnet_e200_seq56.ser"
        saved_model = torch.load(model_path)
        gen_text(saved_model, 1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
