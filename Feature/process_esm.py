import torch
import esm

model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

def get_sequence(filename): 
    dic = {}
    with open(filename) as file:
        for each_line in file:
            if each_line.startswith('>'):
                n = each_line.strip()[1:]
                dic[n] = ''
                continue
            dic[n] += each_line.rstrip()[:1022]
    return dic
pro_seq = get_sequence('pro_seq.txt')
pro_esm_vector_1 = {}  
pro_esm_vector_2 = {}  
from tqdm import tqdm
for k,v in tqdm(pro_seq.items()):
    pro_data = [(k,v)] 
    batch_labels, batch_strs, batch_tokens = batch_converter(pro_data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    pro_esm_vector_1[k] = token_representations
    
    sequence_representations = []
    for i, (_, seq) in enumerate(pro_data):
        sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))
    pro_esm_vector_2[k] = sequence_representations
with open('esm_aa_vector.txt','w') as f:
    for pro,vector in tqdm(pro_esm_vector_1.items()):
        f.write('>')
        f.write(pro)
        f.write('\n')
        vector = vector.numpy()
        
        for i in vector[0]:
            for j in i:
                f.write(str(j))
                f.write(',')
            f.write('\n')
with open('esm_ave_vector.txt','w') as f:
    for pro,vector in tqdm(pro_esm_vector_2.items()):
        f.write('>')
        f.write(pro)
        f.write('\n')
        for i in vector:
            for j in i:
                j = float(j)
                f.write(str(j))
                f.write(',')
        f.write('\n')
                
