import torch

configname = 'epoch_100'
filename = '../mmclassification/work_dirs/imagenet_mixup/'+ configname + '.pth'
savename = filename.replace(configname+'.pth', configname+'_pretrained.pth')


pretrained_dict = torch.load(filename)
new_dict = {}
print(new_dict)
for k, v in pretrained_dict.items():
    print(k)

pass_keys = [
    'head',
]

for k, v in pretrained_dict['state_dict'].items():
    i = k.find('.')
    if i < 0 or any(x in k for x in pass_keys):
        continue
    else:
        k_new = k[i+1:]
        new_dict[k_new] = v

    
    # new_dict[k] = v

    # k_new = k.replace('backbone.', '')
    # # new_dict[k_new] = v
    # k_new = k_new.replace('neck.', '')
    # new_dict[k_new] = v
    # print(k)

for k, v in new_dict.items():
    print(k)
#
# print('==============================================')
# for k, v in pretrained_dict['state_dict'].items():
#     print(k)

pretrained_dict_new = {}
pretrained_dict_new['state_dict'] = new_dict
torch.save(pretrained_dict_new, savename)
print("saved")