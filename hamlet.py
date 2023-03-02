'''
Simple generative text model based on Hamlet
'''
import numpy as np
from scipy.sparse import csr_array, dok_array

with open('hamlet.txt') as f:
    ok_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ?!.'[]"
    ok_chars += ok_chars.lower()
    text = ''.join([c if c in ok_chars else ' ' for c in f.read()])

input_N = 1
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

for i in range(len(tokens) - input_N):
    input_context = ' '.join(tokens[i:i + input_N])
    next_output = tokens[i + input_N]
    input_index = inds_by_input[input_context]
    output_index = inds_by_output[next_output]
    counts[input_index, output_index] += 1

# norms = csr_array(np.sum(counts, axis=1).reshape(-1, 1))
norms = np.sum(counts, axis=1).reshape(-1, 1)
# print(f'{norms.shape = }')
# print(f'{counts.shape = }')
sample_probs = counts / norms # or try https://stackoverflow.com/questions/16043299/substitute-for-numpy-broadcasting-using-scipy-sparse-csc-matrix
sample_probs_uniform = (counts != 0.0).todense().astype(np.float32)  # make dense, sadly
#sample_probs_uniform = counts != 0.0

print(sample_probs.shape)
#print('sparsity = ', 1 - sample_probs.count_nonzero() / sample_probs.size)
print(inputs_by_ind[:3], inputs_by_ind[-3:])
print(sample_probs_uniform.shape)
#print('sparsity = ', 1 - sample_probs_uniform.count_nonzero() / sample_probs_uniform.size)

output = []
seed = np.random.randint(10**8)
print(f'{seed=}, {input_N=}')
np.random.seed(seed)
all_outputs = np.random.choice(inputs_by_ind).split(' ')
# print(all_outputs)
for step in range(1000):
    input_context = ' '.join(all_outputs[-input_N:])
    # print(input_context)
    if np.random.random() < 0.5:  # choose by probability
        p = sample_probs[inds_by_input[input_context]]
    else:  # choose any nonzero probability equally
        p = sample_probs_uniform[inds_by_input[input_context], :]
    p /= p.sum()
    new_output = np.random.choice(outputs_by_ind, p=p)
    all_outputs.append(new_output)
text_out = ' '.join(all_outputs).replace('. ', '.\n').replace('! ', '!\n')
print(text_out)

'''
Fun examples:
seed=39308240, input_N=1
eaten a farm it please you good friends!
How now this play? QUEEN GERTRUDE Nay then That we do pall and is't my story.
March within my bed.
KING CLAUDIUS [Aside] How now a robustious periwig pated fellow might please you Or is king dunks to pass through our bad begins and the pregnant sometimes march? by himself.
KING CLAUDIUS How now!
LAERTES The canker galls the proud revengeful ambitious with all believe it.
ROSENCRANTZ Most generous thoughts That he suffer this rude against our sides and an I see.
Takes the ability of love passing well.
OPHELIA No fairy takes

'''