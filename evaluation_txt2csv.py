# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

INPUT_PATH = 'evaluation_long_train.txt'
OUTPUT_PATH = INPUT_PATH.replace('.txt', '.csv')

# %%
models = []
losses = []
perplexities = []

for line in open(INPUT_PATH, 'r').read().splitlines():
    if line.startswith('model'):
        model = line.split(':')[1].split('/')[-1]
        models.append(model)
    elif line.startswith('loss'):
        loss = float(line.split(':')[1])
        losses.append(loss)
    elif line.startswith('perplexity'):
        perplexity = float(line.split(':')[1])
        perplexities.append(perplexity)
    else:
        continue

df = pd.DataFrame({'model': models, 'loss': losses, 'perplexity': perplexities})
df.to_csv(OUTPUT_PATH)
# %%
sns.set_context('paper')
sns.lineplot(x=range(len(df)), y=df['perplexity'], markers='x')