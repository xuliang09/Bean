from datasketch import MinHashLSHForest, MinHash
import pickle

data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'datasets']
data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents']
data3 = ['minhash', 'is', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents']

dataset = [[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 1., 1.]]
# Create a MinHash LSH Forest with the same num_perm parameter
forest = MinHashLSHForest(num_perm=128)

for i, data in enumerate(dataset):
    m = MinHash(num_perm=128)
    for d in data:
        m.update(str(d).encode('utf8'))
    forest.add(str(i), m)

# IMPORTANT: must call index() otherwise the keys won't be searchable

pickle.dump(forest, open('forest.lsh', 'wb'))
del forest
forest = pickle.load(open('forest.lsh', 'rb'))

forest.index()

# Check for membership using the key
print("1" in forest)
print("2" in forest)

m = MinHash(num_perm=128)
for d in dataset[0]:
    m.update(str(d).encode('utf8'))
# Using m1 as the query, retrieve top 2 keys that have the higest Jaccard
result = forest.query(m, 10)
print("Top 2 candidates", result)