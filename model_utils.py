import torch
from tqdm import tqdm
import math


def evaluate_model(model, batches, prefix='val'):
  model.eval()
  losses_val = []
  with torch.no_grad():
    for batch in tqdm(batches, desc='Evaluating model'):
      batch = {k: v.cuda() for k, v in batch.items()}
      outputs = model(**batch)
      loss_val = outputs.loss
      losses_val.append(loss_val)
    
    # aggregate losses
    losses_val = torch.tensor(losses_val)
    mean_loss_val = torch.mean(losses_val)

    perplexity = compute_perplexity(mean_loss_val)

    # prepare info to log to wandb
    info_to_log = {
      f'{prefix}/perplexity' : perplexity,
      f'{prefix}/loss' : mean_loss_val.item()
    }

  return info_to_log

def compute_perplexity(x):
  try:
    return math.exp(x)
  except OverflowError:
    return float("inf")