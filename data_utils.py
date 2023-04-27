from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import torch_default_data_collator
import os
from collections import defaultdict
import json
import random
import numpy as np
from tqdm import tqdm
import gzip

# TODO: check if it is possible avoiding double conversions between list of dicts and dict of lists (see __explode_chunks())
class DatasetMLM:
  def __init__(self, datapath, tokenizer, chunk_size, n_documents, drop_last = False, mlm_probability = .15,
              whole_word = False, seed = 0, log_filepath = 'log.txt', validation_data=True, docs_to_exclude = set()):
    self.data = self.__get_file_list(datapath, validation_data, n_documents, docs_to_exclude)
    self.tokenizer = tokenizer
    self.chunk_size = chunk_size
    self.drop_last = drop_last
    self.whole_word = whole_word
    if self.whole_word:
      self.data_collator = DataCollatorForWholeWordLanguageModeling(tokenizer=self.tokenizer, mlm_probability=mlm_probability)
    else:
      self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=mlm_probability)
    
    self.seed = seed
    self.__init_generator()
    self.logger = self.__init_logger(log_filepath)
  
  def get_chunks(self, n = 32):
    # collect n chunks
    generator = self.__generate_chunk()
    finished = False
    while not finished:
      chunks = defaultdict(list)
      # chunks = []
      for _ in range(n):
        try:
          chunk = next(generator)
        except StopIteration as e:
          self.logger.write('Iterated over all documents\n')
          self.logger.flush()
          chunk = {}
          finished = True
          break
          # NOTE: uncomment to allow infinite loop
          # yield False
          # self.__init_generator(restart=True)
          # generator = self.__generate_chunk()
          # chunk = next(generator)
        finally:
          for k, v in chunk.items():
            chunks[k].append(v)
          # chunks.append(chunk)
      
      # mask random tokens
      if chunks: 
        chunks = self.data_collator(self.__explode_chunks(chunks)) 
      yield chunks
  
  def __explode_chunks(self, chunks):
    exploded_chunks = []
    for i in range(len(chunks['input_ids'])):
      chunk = {
        k: v[i]
        for k, v in chunks.items()
      }
      exploded_chunks.append(chunk)
    return exploded_chunks

  def __init_logger(self, log_filepath):
    # make directory if not exist
    if '/' in log_filepath:
      log_dir = '/'.join(log_filepath.split('/')[:-1])
      if log_dir not in ['.', '']:
        os.makedirs(log_dir, exist_ok=True)
    # delete old log file
    if os.path.exists(log_filepath):
      os.remove(log_filepath)
    # return new logger
    return open(log_filepath, 'a')

  def __init_generator(self, restart=False):
    if restart:
      self.seed += 1
    random.Random(self.seed).shuffle(self.data)

  def __get_file_list(self, datapath, validation_data, n_documents, docs_to_exclude):
    data = []
    for dirpath, _, filenames in tqdm(os.walk(datapath), desc='Retrieving file paths...'):
      for filename in filenames:
        if n_documents > 0:
          docpath = os.path.join(dirpath, filename)
          if docpath not in docs_to_exclude:
            if validation_data:
              if filename == 'doc_0.json':
                data.append(docpath)
                n_documents -= 1
                break
            else:
              if filename != 'doc_0.json':
                data.append(docpath)
                n_documents -= 1
        else:
          return data
    return data
  
  def __generate_chunk(self):
    # iterate over n_documents
    for docpath in tqdm(self.data, desc='Documents processed'):

      # doc = json.loads(open(docpath, 'r').read())['text']
      with gzip.open(docpath, 'rb') as fd:
        doc = json.load(fd)['text']
      
      tokenized_doc = self.__tokenize(doc)
      chunks = self.__chunk_text(tokenized_doc)
      n_chunks = len(chunks['input_ids'])
      self.logger.write(f'{docpath}: # {n_chunks} chunks\n')
      self.logger.flush()
      
      for i in range(n_chunks):
        chunk = {k : v[i] for k, v in chunks.items()}
        yield chunk

  def __tokenize(self, doc):
    tokenized_doc = self.tokenizer(doc)
    if self.whole_word and self.tokenizer.is_fast:
      tokenized_doc["word_ids"] = tokenized_doc.word_ids()
    return tokenized_doc

  def __chunk_text(self, tokenized_doc):
    # create chunk each output of the tokenizer
    chunks = {
      k: [v[i : i + self.chunk_size] for i in range(0, len(tokenized_doc['input_ids']), self.chunk_size)]
      for k, v in tokenized_doc.items()
    }
    
    # copy labels to keep track of unmasked chunk
    chunks['labels'] = chunks['input_ids'].copy()

    # handle last chunk
    if self.drop_last:
      for k in chunks:
        del chunks[k][-1]
    else:
      last_chunk_length = len(chunks['input_ids'][-1])
      chunks['input_ids'][-1].extend([0] * (self.chunk_size - last_chunk_length))
      chunks['attention_mask'][-1].extend([0] * (self.chunk_size - last_chunk_length))
      chunks['token_type_ids'][-1].extend([0] * (self.chunk_size - last_chunk_length))

    return chunks

class DataCollatorForWholeWordLanguageModeling:
  def __init__(self, tokenizer, mlm_probability = .15):
    self.tokenizer = tokenizer
    self.mlm_probability = mlm_probability

  def __call__(self, examples):
    
    for example in examples:
      word_ids = example.pop("word_ids")

      # Create a map between words and corresponding token indices
      mapping = defaultdict(list)
      current_word_index = -1
      current_word = None
      for idx, word_id in enumerate(word_ids):
        if word_id is not None:
          if word_id != current_word:
            current_word = word_id
            current_word_index += 1
          mapping[current_word_index].append(idx)

      # Randomly mask words
      mask = np.random.binomial(1, self.mlm_probability, (len(mapping),))
      input_ids = example["input_ids"]
      labels = example["labels"]
      new_labels = [-100] * len(labels)
      for word_id in np.where(mask)[0]:
        word_id = word_id.item()
        for idx in mapping[word_id]:
          new_labels[idx] = labels[idx]
          input_ids[idx] = self.tokenizer.mask_token_id
      example["labels"] = new_labels

    return torch_default_data_collator(examples)