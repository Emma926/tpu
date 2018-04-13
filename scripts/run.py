import os
import subprocess

root = os.path.realpath('..')
out_path = os.path.join(root, 'outputs')
model_path= os.path.join(root, 'models/official')
GCS_BUCKET_NAME='tpubenchmarking'

if not os.path.isdir(out_path):
    print('Creating new directory: ' + out_path)
    os.makedirs(out_path)

cmds = {
    'resnet': 'python resnet_main.py'\
    + ' --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet\
    --steps_per_eval=5000\
    --steps_per_checkpoint=100\
    --train_steps=$TRAIN_STEPS\
    --iterations_per_loop=$ITERATIONS\
    --train_batch_size=$BATCH_SIZE\
    --model_dir=gs://' + GCS_BUCKET_NAME + '/resneti/$BATCH_SIZE',

    'densenet':'python densenet_imagenet.py'\
    + ' --alsologtostderr\
    --iterations_per_loop=None\
    --steps_per_checkpoint=100\
    --num_shards=8\
    --mode=\'train\'\
    --train_batch_size=$BATCH_SIZE\
    --train_steps=$TRAIN_STEPS\
    --iterations_per_loop=$ITERATIONS\
    --model_dir=gs://' + GCS_BUCKET_NAME + '/densenet/$BATCH_SIZE\
    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet',

    'mobilenet':'python mobilenet.py' \
    + ' --alsologtostderr\
    --num_shards=8\
    --mode=\'train\'\
    --train_batch_size=$BATCH_SIZE\
    --train_steps=$TRAIN_STEPS\
    --iterations=$ITERATIONS\
    --save_checkpoints_secs=10\
    --model_dir=gs://' + GCS_BUCKET_NAME + '/mobilenet/$BATCH_SIZE\
    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet',

  '':'',
}


for name, cmd in cmds.iteritems():
    name ='resnet'
    cmd = cmds[name]
    batch_size = 128
    iterations = 100
    train_steps = 300

    if not os.path.isdir(os.path.join(out_path, name)):
        print('Creating new directory: ' + os.path.join(out_path, name))
        os.makedirs(os.path.join(out_path, name))

    os.chdir(os.path.join(model_path, name))
    cmd = cmd.replace('$BATCH_SIZE', str(batch_size))
    cmd = cmd.replace('$ITERATIONS', str(iterations))
    cmd = cmd.replace('$TRAIN_STEPS', str(train_steps))
    cmd += ' --use_tpu=True --tpu_name=' + os.uname()[1]
    cmd = " ".join(cmd.split())
    print(cmd)
    file_name = 'batchsize_' + str(batch_size) + '-iteration_' + str(iterations)
    outfile = open(os.path.join(out_path, name, file_name + '.out'), 'w')
    errfile = open(os.path.join(out_path, name, file_name + '.err'), 'w')
    p = subprocess.Popen(cmd.split(' '), stdout=outfile, stderr=errfile)
    p.wait()
    break
