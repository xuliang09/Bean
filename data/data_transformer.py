import random


def read_file(t, dataset):
    if t == 'ent' or t == 'rel':
        file = '/entity2id.txt' if t == 'ent' else '/relation2id.txt'

        id2x, x_total = {}, 0
        with open(dataset + file, 'r', encoding='utf8') as f:
            x_total = int(f.readline().strip())
            for line in f:
                x, id = line.strip().split()
                id2x[int(id)] = x
            return id2x, x_total

    if t == 'train' or t == 'valid' or t == 'test':
        if t == 'train':
            file = '/train2id.txt'
        elif t == 'valid':
            file = '/valid2id.txt'
        else:
            file = '/test2id.txt'

        x_list, x_total = [], 0
        with open(dataset + file, 'r', encoding='utf8') as f:
            x_total = int(f.readline().strip())
            for line in f:
                h, t, r = line.strip().split()
                x_list.append((int(h), int(r), int(t)))
            return x_list, x_total

    raise RuntimeError('parameter 0 not right')


def main(src_dataset, dst_folder, new_train_total=10000, new_test_total=1000, new_valid_total=1000):
    id2ent, ent_total = read_file('ent', src_dataset)
    id2rel, rel_total = read_file('rel', src_dataset)
    train_list, train_total = read_file('train', src_dataset)
    test_list, test_total = read_file('test', src_dataset)
    valid_list, valid_total = read_file('valid', src_dataset)

    new_train_list = [train_list[i] for i in random.sample([i for i in range(train_total)], new_train_total)]

    new_entity_list = []
    new_entity_list += [h for h, r, t in new_train_list]
    new_entity_list += [t for h, r, t in new_train_list]
    new_entity_list = list(set(new_entity_list))

    new_relation_list = list({r for h, r, t in new_train_list})

    new_test_list = [(h, r, t) for h, r, t in test_list if
                     h in new_entity_list and t in new_entity_list and r in new_relation_list]
    if len(new_test_list) > new_test_total:
        new_test_list = new_test_list[:new_test_total]
    else:
        new_test_total = len(new_test_list)

    new_valid_list = [(h, r, t) for h, r, t in valid_list if
                      h in new_entity_list and t in new_entity_list and r in new_relation_list]
    if len(new_valid_list) > new_valid_total:
        new_valid_list = new_valid_list[:new_valid_total]
    else:
        new_valid_total = len(new_valid_list)

    ent_old2new = {ent_id: index for index, ent_id in enumerate(new_entity_list)}
    rel_old2new = {rel_id: index for index, rel_id in enumerate(new_relation_list)}

    with open(dst_folder + '/entity2id.txt', 'w', encoding='utf8') as f:
        f.write(str(len(new_entity_list)) + '\n')
        for old_id in new_entity_list:
            f.write(id2ent[old_id] + '\t' + str(ent_old2new[old_id]) + '\n')
    with open(dst_folder + '/relation2id.txt', 'w', encoding='utf8') as f:
        f.write(str(len(new_relation_list)) + '\n')
        for old_id in new_relation_list:
            f.write(id2rel[old_id] + '\t' + str(rel_old2new[old_id]) + '\n')

    with open(dst_folder + '/train2id.txt', 'w', encoding='utf8') as f:
        f.write(str(new_train_total) + '\n')
        for h, r, t in new_train_list:
            f.write(str(ent_old2new[h]) + '\t' + str(ent_old2new[t]) + '\t' + str(rel_old2new[r]) + '\n')
    with open(dst_folder + '/test2id.txt', 'w', encoding='utf8') as f:
        f.write(str(new_test_total) + '\n')
        for h, r, t in new_test_list:
            f.write(str(ent_old2new[h]) + '\t' + str(ent_old2new[t]) + '\t' + str(rel_old2new[r]) + '\n')
    with open(dst_folder + '/valid2id.txt', 'w', encoding='utf8') as f:
        f.write(str(new_valid_total) + '\n')
        for h, r, t in new_valid_list:
            f.write(str(ent_old2new[h]) + '\t' + str(ent_old2new[t]) + '\t' + str(rel_old2new[r]) + '\n')
