import os
import subprocess

out_path = '/home/emma/tpu/outputs'
model_path='/home/emma/tpu/models/official'
GCS_BUCKET_NAME='tpubenchmarking'


cmds = {'resnet': 'python ' + os.path.join(model_path,'resnet/resnet_main.py') \
  + ' --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet \
  --model_dir=gs://' + GCS_BUCKET_NAME + '/resnet',
  '':'',
  '':'',
}

for name, cmd in cmds.iteritems():
  name ='resnet'
  cmd = cmds[name]
  cmd += ' --use_tpu=True --train_steps=300 --tpu_name=' + os.uname()[1]
  print(cmd)
  outfile = open(os.path.join(out_path, name, 'out'), 'w')
  errfile = open(os.path.join(out_path, name, 'err'), 'w')
  subprocess.call(cmd.split(' '), stdout=outfile, stderr=errfile)
  break
