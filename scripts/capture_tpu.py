# https://cloud.google.com/tpu/docs/cloud-tpu-tools
# Install capture_tpu_profile
#(vm)$ pip freeze | grep cloud-tpu-profiler
#(vm)$ sudo pip install --upgrade "cloud-tpu-profiler==1.5.2"


import os
import sys
import time

cmd = 'capture_tpu_profile --tpu_name=' + os.uname()[1] + ' --duration_ms=60000 --logdir=gs://tpubenchmarking/tpu_trace'

folder = sys.argv[1]

time.sleep(60)
count = 0

os.system('gsutil rm -r gs://tpubenchmarking/tpu_trace/' + folder)

while(1):
  os.system(cmd + '/' + folder) 
  count += 1
  time.sleep(60)
 
