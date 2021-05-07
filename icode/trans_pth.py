import torch

filename = 'moco_v1_200ep_pretrain.pth.tar'
savename = filename.split('.')
savename = savename[0] + '_rename.pth'
pretrained_dict = torch.load(filename)
new_dict = {}
print(new_dict)
for k, v in pretrained_dict.items():
    print(k)

for k, v in pretrained_dict['state_dict'].items():
    k_new = k.replace('module.encoder_q.', '')
    new_dict[k_new] = v
    print(k)

for k, v in new_dict.items():
    print(k)

pretrained_dict_new = {}
pretrained_dict_new['state_dict'] = new_dict
torch.save(pretrained_dict_new, savename)
print("saved")