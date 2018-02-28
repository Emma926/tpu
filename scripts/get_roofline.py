import os
import math
import numpy as np

path = '/home/emma/tpu/outputs'

files = os.listdir(path)
label_all = []
param_all = []
intensity_all = []
flops_all = []

for f in files:
  out_f = os.path.join(path, f, 'out')
  os.system('grep _TFProfRoot ' + out_f + ' > tmp')
  tmp = open('tmp', 'r')
  if os.stat('tmp').st_size == 0:
    continue
  for line in tmp:
    if 'params' in line:
      line = line.split('/')[-1].split(' ')[0]
      if line[-1] == 'k':
        param = float(line.strip('k')) * 1e3
      if line[-1] == 'm':
        param = float(line.strip('m')) * 1e6
      if line[-1] == 'b':
        param = float(line.strip('b')) * 1e9
    elif 'flops' in line:
      line = line.split('/')[-1].split(' ')[0]
      if line[-1] == 'k':
        flops = float(line.strip('k')) * 1e3
      if line[-1] == 'm':
        flops = float(line.strip('m')) * 1e6
      if line[-1] == 'b':
        flops = float(line.strip('b')) * 1e9

  err_f = os.path.join(path, f.replace('.out', '.err'))
  err_f = os.path.join(path, f, 'err')
  print(err_f)
  print('grep \"global_step/sec\" ' + err_f + ' > tmp')
  os.system('grep \"global_step/sec\" ' + err_f + ' > tmp')
  tmp = open('tmp', 'r')
  speed = []
  for line in tmp:
    if "global_step/sec" in line:
      speed.append(float(line.strip('\n').split(' ')[-1]))
    else:
      continue
  step_per_sec = np.mean(speed)
  
  flops_per_sec = flops * step_per_sec / 1e9 
  if math.isnan(flops_per_sec):
    continue
  flops_all.append(flops_per_sec)
  param_all.append(param)
  intensity_all.append(flops/(param/4.0))
  label_all.append(f)

print 'labels = ' + str(label_all)
#print param_all
print 'intensity = ' + str(intensity_all)
print 'flops = ' + str(flops_all)

