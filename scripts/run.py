import os
import subprocess

out_path = '/home/emma/tpu/outputs'
model_path='/home/emma/tpu/models/official'
GCS_BUCKET_NAME='tpubenchmarking'


cmds = {'resnet': 'python resnet_main.py'\
  + ' --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet \
  --model_dir=gs://' + GCS_BUCKET_NAME + '/resnet',

  'densenet':'python densenet_imagenet.py'\
  + ' --alsologtostderr\
   --num_shards=8\
   --batch_size=1024\
 --model_dir=gs://' + GCS_BUCKET_NAME + '/densenet\
  --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet',

  'mobilenet':'python mobilenet.py' \
  + ' --alsologtostderr\
  --num_shards=8\
  --train_batch_size=1024\
  --model_dir=gs://' + GCS_BUCKET_NAME + '/mobilenet\
  --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet',

  '':'',
}

for name, cmd in cmds.iteritems():
  name ='mobilenet'
  cmd = cmds[name]

  os.chdir(os.path.join(model_path, name))
  cmd += ' --use_tpu=True --train_steps=3000 --tpu_name=' + os.uname()[1]
  print(cmd)
  outfile = open(os.path.join(out_path, name, 'out'), 'w')
  errfile = open(os.path.join(out_path, name, 'err'), 'w')
  subprocess.call(cmd.split(' '), stdout=outfile, stderr=errfile)
  break
