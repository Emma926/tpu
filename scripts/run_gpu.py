import os
import subprocess

root = os.path.realpath('..')
out_path = os.path.join(root, 'outputs')
model_path= os.path.join(root, 'models/local')
GCS_BUCKET_NAME='tpubenchmarking'

if not os.path.isdir(out_path):
    print('Creating new directory: ' + out_path)
    os.makedirs(out_path)

# densenet does not run with this script for now
cmds = {

#    'resnet_float16_fake': ('resnet_float16_fake', 'python resnet_main.py'\
#    + ' --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet\
#    --steps_per_eval=5000\
#    --train_steps=$TRAIN_STEPS\
#    --iterations_per_loop=$ITERATIONS\
#    --train_batch_size=$BATCH_SIZE\
#    --num_parallel_calls=192\
#    --mode=train\
#    --use_transpose=1\
#    --model_dir=gs://$GCS_BUCKET_NAME/tmp'),
#
#    'densenet_float16_fake': ('densenet_float16_fake', 'python densenet_imagenet.py'\
#    + ' --alsologtostderr\
#    -steps_per_checkpoint=100\
#    --num_shards=8\
#    --mode=train\
#    --train_batch_size=$BATCH_SIZE\
#    --train_steps=$TRAIN_STEPS\
#    --iterations_per_loop=$ITERATIONS\
#    --model_dir=gs://$GCS_BUCKET_NAME/tmp\
#    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet'),

#    'mobilenet_float16_fake':('mobilenet_float16', 'python mobilenet.py' \
#    + ' --alsologtostderr\
#    --num_shards=8\
#    --mode=train\
#    --use_data=fake\
#    --train_batch_size=$BATCH_SIZE\
#    --train_steps=$TRAIN_STEPS\
#    --iterations=$ITERATIONS\
#    --save_checkpoints_secs=10\
#    --model_dir=gs://$GCS_BUCKET_NAME/tmp\
#    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet'),

    'retinanet_float16_fake':('retinanet_float16_fake', 'python retinanet_main.py'\
    + ' --train_batch_size=$BATCH_SIZE\
    --training_file_pattern=gs://$GCS_BUCKET_NAME/coco/train-* \
    --resnet_checkpoint=gs://cloud-tpu-artifacts/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603 \
     --model_dir=gs://$GCS_BUCKET_NAME/tmp\
    --iterations_per_loop=$ITERATIONS\
    --train_steps=$TRAIN_STEPS\
    --hparams=image_size=640 \
    --num_examples_per_epoch=6400 \
    --num_epochs=1'),

#    'squeezenet_float16_fake':('squeezenet_float16_fake','python squeezenet_main.py' \
#    + ' --alsologtostderr\
#    --num_shards=8\
#    --num_evals=0\
#    --batch_size=$BATCH_SIZE\
#    --train_steps=$TRAIN_STEPS\
#    --iterations=$ITERATIONS\
#    --save_checkpoints_secs=10\
#    --model_dir=gs://$GCS_BUCKET_NAME/tmp\
#    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet'),
}

configs = []
#for bs in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
#    for it in [100, 1000, 10000]:
#        configs.append((bs, it))

configs = [(1024, 100, 300), (1024, 1000, 5000)]
configs = [(1024, 1000, 5000)]
configs = [(256, 1000, 500)]

for config in configs:
    for name, (directory, cmd) in cmds.items():
        (batch_size, iterations, train_steps) = config
        #if not os.path.isdir(os.path.join(out_path, name)):
        #    print('Creating new directory: ' + os.path.join(out_path, name))
        #    os.makedirs(os.path.join(out_path, name))
        if 'retina' in name:
          batch_size = 16
          iterations = 100
	  train_steps = 500

        os.system('gsutil rm -r gs://' + GCS_BUCKET_NAME + '/tmp')
        file_name = name + '-batchsize_' + str(batch_size) + '-iteration_' + str(iterations) + '-trainsteps_' + str(train_steps)
        #os.system('grep \"global_step/sec\" ' + os.path.join(out_path, name, file_name + '.err') + ' > tmp')
        #if not os.stat('tmp').st_size == 0:
        #    continue

        os.chdir(os.path.join(model_path, directory))
        if not 'BATCH_SIZE' in cmd:
          print(name, '\'s cmd does not have BATCH_SIZE.')
          continue
        if not 'ITERATIONS' in cmd:
          print(name, '\'s cmd does not have ITERATIONS.')
          continue
        if not 'TRAIN_STEPS' in cmd:
          print(name, '\'s cmd does not have TRAIN_STEPS.')
          continue
        
        cmd = cmd.replace('$GCS_BUCKET_NAME', GCS_BUCKET_NAME)
        cmd = cmd.replace('$BATCH_SIZE', str(batch_size))
        cmd = cmd.replace('$ITERATIONS', str(iterations))
        cmd = cmd.replace('$TRAIN_STEPS', str(train_steps))
        cmd = cmd.replace('$MODEL_DIR', file_name)
        cmd += ' --use_tpu=False'
        cmd = " ".join(cmd.split())

        print(name, os.path.join(out_path, file_name + '.err'))
        print(cmd)

        outfile = open(os.path.join(out_path, file_name + '.out'), 'w')
        errfile = open(os.path.join(out_path, file_name + '.err'), 'w')
        p = subprocess.Popen(cmd.split(' '), stdout=outfile, stderr=errfile)
        p.wait()
