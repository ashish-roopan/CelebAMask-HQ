import os
import sys
import cv2
import time
import numpy as np
from PIL import Image




import torch
from torchvision.utils import save_image

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from data.base_dataset import BaseDataset, get_params, get_transform, normalize






os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
#model = Model(config)
opt = TestOptions().parse(save=False)
print(opt)
print(type(opt))
print('opt,model',opt.model)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
model = create_model(opt)


mask=cv2.imread('samples/0.png')
mask_m=cv2.imread('../../gdrive/My Drive/faceswap/mask3.png')
mat_img = Image.open('samples/0.jpg')
img = mat_img.copy()

params = get_params(opt, (512,512))
transform_mask = get_transform(opt, params, method=Image.NEAREST, normalize=False, normalize_mask=True)
transform_image = get_transform(opt, params)



mask = transform_mask(Image.fromarray(np.uint8(mask))) 
mask_m = transform_mask(Image.fromarray(np.uint8(mask_m)))
img = transform_image(img)



start_t = time.time()
generated = model.inference(torch.FloatTensor([mask_m.numpy()]), torch.FloatTensor([mask.numpy()]), torch.FloatTensor([img.numpy()]))   
end_t = time.time()
print('inference time : {}'.format(end_t-start_t))
#save_image((generated.data[0] + 1) / 2,'./results/1.jpg')
result = generated.permute(0, 2, 3, 1)
result = result.cpu().detach().numpy()
result = (result + 1) * 127.5
result = np.asarray(result[0,:,:,:], dtype=np.uint8)
result= cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
cv2.imwrite('gen.jpg',result)
print(result)
