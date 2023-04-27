# %%
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
from data_utils import DatasetMLM
from model_utils import compute_perplexity, evaluate_model
import torch
from torch.optim import AdamW
from argparse import ArgumentParser
import wandb
import torch
# torch.use_deterministic_algorithms(True)
import random
import numpy as np

parser = ArgumentParser('Train Masked Language Model')

parser.add_argument('-b', '--bert', type=str, default='dlicari/ita-legal-bert', help='BERT version name for AutoModel')
parser.add_argument('--ntrain', type=int, help='Number of documents to use for training')
parser.add_argument('--nval', type=int, help='Number of documents to use for validation')
parser.add_argument('--ckptrate', type=int, default=100, help='Save every N batch')
parser.add_argument('--valrate', type=int, default=100, help='Validate and log every N batch')
parser.add_argument('--batch', type=int, default=32, help='Batch size')
parser.add_argument('--chunk', type=int, default=128, help='Chunk size')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--mlmprob', type=float, default=.15, help='MLM probability')
parser.add_argument('-w','--wholeword', action='store_true', help='Mask whole word instead of token')
parser.add_argument('-s', '--seed', type=int, default=False, help='Random seed')
parser.add_argument('--droplast', action='store_true', help='Drop last chunk of the document instead of add padding')
parser.add_argument('-d', '--data', help='Data directory path. Make sure to have doc_0.json files in each subdirectory, since they are used as validation documents')
parser.add_argument('--user', type=str, help='wandb user')
parser.add_argument('--project', type=str, help='wandb project')
parser.add_argument('--name', type=str, default='mlm', help='Model ckpt name')
parser.add_argument('-v', '--verbose', action='store_true', help='Print training info')
args = parser.parse_args()

BERT_VERSION = args.bert
N_DOCUMENTS_TRAIN = args.ntrain
N_DOCUMENTS_VAL = args.nval
CKPT_RATE = args.ckptrate
VAL_RATE = args.valrate
BATCH_SIZE = args.batch
CHUNK_SIZE = args.chunk
LR = args.lr
MLM_PROBABILITY = args.mlmprob
WHOLE_WORD = args.wholeword
SEED = args.seed
DROP_LAST = args.droplast
DATA_DIR = args.data
WANDB_USER = args.user
WANDB_PROJECT = args.project
WANDB_KEY = wandb.login()
VERBOSE = args.verbose
MODEL_NAME = args.name
MODEL_DIR = f'./model/{MODEL_NAME}'
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_" + '{}.ckpt')
LOG_DIR = f'./log/{MODEL_NAME}'
LOG_TRAIN_PATH = os.path.join(LOG_DIR, 'log_train_dataloader.txt')
LOG_VAL_PATH = os.path.join(LOG_DIR, 'log_val_dataloader.txt')

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
  

# %%
# PREPARE MODEL, TOKENIZER, OPTIMIZER
print(f'Loading {BERT_VERSION} model...')
tokenizer = AutoTokenizer.from_pretrained(BERT_VERSION)
model = AutoModelForMaskedLM.from_pretrained(BERT_VERSION)
model = model.cuda()
# TODO: use defalut parameters?
optimizer = AdamW(model.parameters(), lr=LR)

# %%
# PREPARE DATASETS
print(f'Preparing dataset from {DATA_DIR}')
# train
train_dataset = DatasetMLM(datapath=DATA_DIR, tokenizer=tokenizer, chunk_size=CHUNK_SIZE, n_documents=N_DOCUMENTS_TRAIN,
                    drop_last=DROP_LAST, mlm_probability=MLM_PROBABILITY, whole_word=WHOLE_WORD, seed=SEED,
                    log_filepath=LOG_TRAIN_PATH, validation_data=False)

# val
val_dataset = DatasetMLM(datapath=DATA_DIR, tokenizer=tokenizer, chunk_size=CHUNK_SIZE, n_documents=N_DOCUMENTS_VAL,
                    drop_last=DROP_LAST, mlm_probability=MLM_PROBABILITY, whole_word=WHOLE_WORD, seed=SEED,
                    log_filepath=LOG_VAL_PATH, validation_data=True)

print('Loading validation batches...')
val_dataset_batches = []
generator = val_dataset.get_chunks(BATCH_SIZE)
batch_val = next(generator)
while batch_val:
  val_dataset_batches.append(batch_val)
  try:
    batch_val = next(generator)
  except StopIteration as e:
    batch_val = False
  
val_dataset_batches


# %%
# INIT WANDB
run = wandb.init(
    project=WANDB_PROJECT,
    name=MODEL_NAME,
    config=vars(args))


# %%
# TRAIN MODEL
generator = train_dataset.get_chunks(BATCH_SIZE)
batch_train = next(generator)
losses_train = []
step = 0
# iterate until N_DOCUMENTS is reached
while batch_train:
  # Training
  model.train()
  batch_train = {k: v.cuda() for k, v in batch_train.items()}
  outputs = model(**batch_train)
  loss_train = outputs.loss
  losses_train.append(loss_train.item())
  loss_train.backward()
  optimizer.step()
  optimizer.zero_grad()

  # Evaluation and log
  if step % VAL_RATE == 0:
    # get validation set metrics
    info_to_log = evaluate_model(model, val_dataset_batches)
    
    # NOTE: in the first step it is only one batch
    loss_train = torch.tensor(losses_train).mean().item()
    losses_train = []
    
    # add train metrics
    perplexity_train = compute_perplexity(loss_train)
    info_to_log_train = {
        'train/perplexity' : perplexity_train,
        'train/loss' : loss_train,
        'utils/real_step' : step,
        'utils/chunks_seen' : step * BATCH_SIZE
      }
    info_to_log.update(info_to_log_train)
    wandb.log(info_to_log)
    
    if VERBOSE:
      summary = '\n'.join([f'{k} : {round(v, 5)}' for k, v in info_to_log.items()])
      print(summary)

  # save model
  if step % CKPT_RATE == 0:
    if VERBOSE:
      print('Saving model at step', step)
    
    model_path = MODEL_PATH.format(step // CKPT_RATE)
    torch.save(model, model_path)
  
  try:
    batch_train = next(generator)
    step += 1
  except StopIteration as e:
    batch_train = False


# save last model
if step % CKPT_RATE != 0:
  print('Saving final model at step', step)
  model_path = MODEL_PATH.format('final')
  torch.save(model, MODEL_PATH)

# evaluate last step
if step % VAL_RATE != 0:
  info_to_log = evaluate_model(model, val_dataset_batches)
    
  loss_train = torch.tensor(losses_train).mean().item()
  losses_train = []

  perplexity_train = compute_perplexity(loss_train)
  info_to_log_train = {
      'train/perplexity' : perplexity_train,
      'train/loss' : loss_train,
      'utils/real_step' : step,
      'utils/chunks_seen' : step * BATCH_SIZE
    }
  info_to_log.update(info_to_log_train)
  wandb.log(info_to_log)

# %%
