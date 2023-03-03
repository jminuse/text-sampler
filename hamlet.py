'''
Simple generative text model based on Shakespeare
'''
import numpy as np
from scipy.sparse import dok_array
from tqdm import tqdm

# https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt
with open('shakespeare.txt') as f:
    text = f.read()
ok_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ?!.'[]"
ok_chars += ok_chars.lower()
text = ''.join([c if c in ok_chars else ' ' for c in text])

input_N = 2  # input_N = 3 tends to simply quote
tokens = text.split()
inputs = [' '.join(tokens[i:i + input_N]) for i in range(0, len(tokens) - input_N)]  # all input contexts
inputs_by_ind = list(sorted(set(inputs)))
outputs_by_ind = list(sorted(set(tokens)))  # all possible outputs
print('Unique inputs:', len(inputs_by_ind))
print('Unique outputs:', len(outputs_by_ind))
# could use scipy.sparse.dok_array for counts to save memory
counts = dok_array((len(inputs_by_ind), len(outputs_by_ind)), dtype=np.float32)
inds_by_input = {w:i for i, w in enumerate(inputs_by_ind)}
inds_by_output = {w:i for i, w in enumerate(outputs_by_ind)}
totals_by_input_ind = np.zeros(len(inputs_by_ind))
print('First and last inputs:', inputs_by_ind[:3], inputs_by_ind[-3:])

for i in tqdm(range(len(tokens) - input_N)):
    input_context = ' '.join(tokens[i:i + input_N])
    next_output = tokens[i + input_N]
    input_index = inds_by_input[input_context]
    output_index = inds_by_output[next_output]
    counts[input_index, output_index] += 1
    totals_by_input_ind[input_index] += 1

print(f'{counts.shape = }')
print(f'{counts.count_nonzero() = }')

sample_probs = dok_array((len(inputs_by_ind), len(outputs_by_ind)), dtype=np.float32)
count_nonzero_by_input_ind = np.zeros(len(inputs_by_ind))
for input_index, output_index in tqdm(zip(*counts.nonzero())):
    sample_probs[input_index, output_index] = counts[input_index, output_index] / totals_by_input_ind[input_index]
    count_nonzero_by_input_ind[input_index] += 1
    
sample_probs_uniform = dok_array((len(inputs_by_ind), len(outputs_by_ind)), dtype=np.float32)
for input_index, output_index in tqdm(zip(*counts.nonzero())):
    sample_probs_uniform[input_index, output_index] = 1.0 / count_nonzero_by_input_ind[input_index]

print(f'{sample_probs.shape = }')
print(f'{sample_probs.count_nonzero() = }')

output = []
seed = np.random.randint(10**8)
print(f'{seed=}, {input_N=}')
np.random.seed(seed)
all_outputs = np.random.choice(inputs_by_ind).split(' ')
# print(all_outputs)
sample_N = 200
for step in range(sample_N):
    input_context = ' '.join(all_outputs[-input_N:])
    # print(input_context)
    input_index = [inds_by_input[input_context]]
    if np.random.random() < 0.5:  # choose by probability
        p = sample_probs[input_index, :]
    else:  # choose any nonzero probability equally (often will switch topics)
        p = sample_probs_uniform[input_index, :]
    p = p.todense().flatten()  # assuming np.random.choice wants dense input
    p /= p.sum()  # make sure probabilities are fully normalized
    new_output = np.random.choice(outputs_by_ind, p=p)
    all_outputs.append(new_output)
# add line breaks at the end of sentences
text_out = ' '.join(all_outputs).replace('. ', '.\n').replace('! ', '!\n')
print('=' * 40)
print(text_out)
print('=' * 40)
