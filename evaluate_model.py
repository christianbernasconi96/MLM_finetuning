# %%
from data_utils import DatasetMLM
from model_utils import compute_perplexity
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import random
import torch
import os
# torch.use_deterministic_algorithms(True)

MODEL_NAME = 'mlm_sentenze'
MODEL_DIR = f'./model/{MODEL_NAME}'
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_" + '{}.ckpt')
MODELS = [str(x) for x in range(9)] + ['final']
N_DOCUMENTS = 20000
LOG_DIR = f'./log/{MODEL_NAME}'
LOG_TRAIN_PATH = os.path.join(LOG_DIR, 'log_train_dataloader.txt')
LOG_TEST_PATH = os.path.join(LOG_DIR, 'log_test_dataloader.txt')
BERT_VERSION = 'dlicari/Italian-Legal-BERT'
BATCH_SIZE = 64
CHUNK_SIZE = 200
MLM_PROBABILITY = .15
SEED = 0
WHOLE_WORD = False
DROP_LAST = False
DATA_DIR = '/home/rpozzi/temp_archive/archives/sentenze_pulite_json/'
OUT_DIR = f'./evaluation/{MODEL_NAME}'
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, 'evaluation.{}')


torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
# %%
docs_to_exclude = [] 
with open(LOG_TRAIN_PATH) as f:
  for line in f.readlines():
    docs_to_exclude.append(line.split(': #')[0].strip())
print('# docs to exclude:', len(docs_to_exclude))
# %%
# prepare dataset
tokenizer = AutoTokenizer.from_pretrained(BERT_VERSION)
test_dataset = DatasetMLM(datapath=DATA_DIR, tokenizer=tokenizer, chunk_size=CHUNK_SIZE,
                          n_documents=N_DOCUMENTS, drop_last=DROP_LAST, mlm_probability=MLM_PROBABILITY,
                          whole_word=WHOLE_WORD, seed=SEED, log_filepath=LOG_TEST_PATH, validation_data=False)

# %%

out_path_txt = OUT_PATH.format('txt')
out_path_csv = OUT_PATH.format('csv')
with open(out_path_txt, 'w') as logger:
  models_df = []
  losses_df = []
  perplexities_df = []
  for model_suffix in MODELS:
    # load model
    model_path = MODEL_PATH.format(model_suffix)
    model = torch.load(model_path).cuda()
    model.eval()

    # evaluate model
    generator = test_dataset.get_chunks(BATCH_SIZE)
    batch = next(generator)
    losses = []
    # perplexities = []
    with torch.no_grad():
      while batch:
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss.item()
        losses.append(loss)
        # perplexity = compute_perplexity(loss)
        # perplexities.append(perplexity)
        try:
          batch = next(generator)
        except StopIteration as e:
          batch = False

    # compute metrics
    loss_mean = torch.tensor(losses).mean().item()
    perplexity_mean = compute_perplexity(loss_mean)

    # log to txt
    str_to_log = '\n'.join([f'model: {model_path}', f'loss: {round(loss_mean, 5)}', f'perplexity: {round(perplexity_mean, 5)}', '-'*10, '\n'])
    logger.write(str_to_log)

    # collect metrics for csv
    models_df.append(model_path)
    losses_df.append(loss_mean)
    perplexities_df.append(perplexity_mean)

# create df and save to csv
df_dict = {
  'model': models_df,
  'loss': losses_df,
  'perplexity': perplexities_df
}
df = pd.DataFrame(df_dict)
df.to_csv(out_path_csv, index=False)

# %%
