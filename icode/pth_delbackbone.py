import torch

filename = '../work_dirs/carafed_coco_faster_r50_1x_3_kernelexp/epoch_12.pth'
# filename = '../work_dirs/carafed_coco_faster_r50_1x/epoch_12.pth'
savename = filename.replace('epoch_12.pth', 'pretrained.pth')

pretrained_dict = torch.load(filename)
new_dict = {}
print(new_dict)
for k, v in pretrained_dict.items():
    print(k)

pass_keys = [
    'neck.lateral',
    'rpn_head.rpn',
    'roi_head.bbox_head',
]

for k, v in pretrained_dict['state_dict'].items():
    i = k.find('.')
    if i >= 0 and all(x not in k for x in pass_keys):
        k_new = k[i+1:]
        new_dict[k_new] = v
    else:
        new_dict[k] = v
    
    # new_dict[k] = v

    # k_new = k.replace('backbone.', '')
    # # new_dict[k_new] = v
    # k_new = k_new.replace('neck.', '')
    # new_dict[k_new] = v
    # print(k)

for k, v in new_dict.items():
    print(k)

print('==============================================')
# for k, v in pretrained_dict['state_dict'].items():
#     print(k)

pretrained_dict_new = {}
pretrained_dict_new['state_dict'] = new_dict
torch.save(pretrained_dict_new, savename)
print("saved")