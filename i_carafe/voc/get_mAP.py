
path = './slurm-1014.out'

def get_info(path):
    with open(path) as file:
        mAP = []
        for row in file:
            words = row.split(' ')
            words = [x for x in words if x not in ['', '|']]
            # print(words)
            if words[0] == 'mAP':
                mAP.append(words[1])
        return mAP

mAP = get_info(path)
mAP = '\t'.join(mAP)
print(mAP)