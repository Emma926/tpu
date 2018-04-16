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
    --model_dir=gs://$GCS_BUCKET_NAME/resneti/$MODEL_DIR',

    'densenet':'python densenet_imagenet.py'\
    + ' --alsologtostderr\
    --iterations_per_loop=None\
    --steps_per_checkpoint=100\
    --num_shards=8\
    --mode=\'train\'\
    --train_batch_size=$BATCH_SIZE\
    --train_steps=$TRAIN_STEPS\
    --iterations_per_loop=$ITERATIONS\
    --model_dir=gs://$GCS_BUCKET_NAME/densenet/$MODEL_DIR\
    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet',

    'mobilenet':'python mobilenet.py' \
    + ' --alsologtostderr\
    --num_shards=8\
    --mode=\'train\'\
    --train_batch_size=$BATCH_SIZE\
    --train_steps=$TRAIN_STEPS\
    --iterations=$ITERATIONS\
    --save_checkpoints_secs=10\
    --model_dir=gs://$GCS_BUCKET_NAME/mobilenet/$MODEL_DIR\
    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet',

    'retinanet':'python lgMain_mp.py'\
    + ' --train_batch_size=$BATCH_SIZE\
    --training_file_pattern==$GCS_BUCKET_NAME/coco/train-* \
    --resnet_checkpoint=${RESNET_CHECKPOINT} \
    --model_dir=gs://$GCS_BUCKET_NAME/retinanet/$MODEL_DIR\
    --hparams=image_size=640 \
    --num_examples_per_epoch=6400 \
    --num_epochs=1',
    
    'squeezenet':'python squeezenet_main.py' \
    + ' --alsologtostderr\
    --num_shards=8\
    --optimizer=\'rmsprop\'\
    --num_evals=10000\
    --train_batch_size=$BATCH_SIZE\
    --train_steps=$TRAIN_STEPS\
    --iterations=$ITERATIONS\
    --save_checkpoints_secs=10\
    --model_dir=gs://$GCS_BUCKET_NAME/squeezenet/$MODEL_DIR\
    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet',

}


for name, cmd in cmds.iteritems():
    name ='squeezenet'
    cmd = cmds[name]
    batch_size = 128
    iterations = 100
    train_steps = 300

    if not os.path.isdir(os.path.join(out_path, name)):
        print('Creating new directory: ' + os.path.join(out_path, name))
        os.makedirs(os.path.join(out_path, name))

    file_name = 'batchsize_' + str(batch_size) + '-iteration_' + str(iterations)

    os.chdir(os.path.join(model_path, name))
    cmd = cmd.replace('$GCS_BUCKET_NAME', GCS_BUCKET_NAME)
    cmd = cmd.replace('$BATCH_SIZE', str(batch_size))
    cmd = cmd.replace('$ITERATIONS', str(iterations))
    cmd = cmd.replace('$TRAIN_STEPS', str(train_steps))
    cmd = cmd.replace('$MODEL_DIR', file_name)
    cmd += ' --use_tpu=True --tpu_name=' + os.uname()[1]
    cmd = " ".join(cmd.split())
    print(cmd)
    outfile = open(os.path.join(out_path, name, file_name + '.out'), 'w')
    errfile = open(os.path.join(out_path, name, file_name + '.err'), 'w')
    p = subprocess.Popen(cmd.split(' '), stdout=outfile, stderr=errfile)
    p.wait()
    break
