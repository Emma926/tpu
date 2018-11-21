import os
import subprocess

root = os.path.realpath('..')
out_path = os.path.join(root, 'outputs')
model_path= os.path.join(root, 'models/local')
GCS_BUCKET_NAME='tpubenchmarking'
tmp_dir = 'gs://' + GCS_BUCKET_NAME + '/tmp'

if not os.path.isdir(out_path):
    print('Creating new directory: ' + out_path)
    os.makedirs(out_path)

# densenet does not run with this script for now
cmds = {
#    'resnet':('resnet', 'python resnet_main.py'\
#    + ' --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet\
#    --steps_per_eval=5000\
#    --steps_per_checkpoint=100\
#    --train_steps=$TRAIN_STEPS\
#    --iterations_per_loop=$ITERATIONS\
#    --train_batch_size=$BATCH_SIZE\
#    --tpu_name=$TPU_NAME\
#    --model_dir=$MODEL_DIR'),

#    'resnet_fake':('resnet_fake', 'python resnet_main.py'\
#    + ' --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet\
#    --steps_per_eval=5000\
#    --steps_per_checkpoint=100\
#    --train_steps=$TRAIN_STEPS\
#    --iterations_per_loop=$ITERATIONS\
#    --train_batch_size=$BATCH_SIZE\
#    --tpu_name=$TPU_NAME\
#    --model_dir=$MODEL_DIR'),

#    'resnet_bfloat16':('resnet_bfloat16', 'python resnet_main.py'\
#    + ' --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet\
#    --steps_per_eval=5000\
#    --steps_per_checkpoint=100\
#    --train_steps=$TRAIN_STEPS\
#    --iterations_per_loop=$ITERATIONS\
#    --train_batch_size=$BATCH_SIZE\
#    --tpu_name=$TPU_NAME\
#    --model_dir=$MODEL_DIR'),
#
    'resnet_bfloat16_fake': ('resnet_bfloat16_fake', 'python resnet_main.py'\
    + ' --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet\
    --steps_per_eval=5000\
    --train_steps=$TRAIN_STEPS\
    --iterations_per_loop=$ITERATIONS\
    --train_batch_size=$BATCH_SIZE\
    --num_parallel_calls=192\
    --mode=train\
    --use_transpose=1\
    --tpu_name=$TPU_NAME\
    --model_dir=$MODEL_DIR'),

#    'densenet': ('densenet', 'python densenet_imagenet.py'\
#    + ' --alsologtostderr\
#    --steps_per_checkpoint=100\
#    --num_shards=8\
#    --mode=train\
#    --train_batch_size=$BATCH_SIZE\
#    --train_steps=$TRAIN_STEPS\
#    --iterations_per_loop=$ITERATIONS\
#    --model_dir=$MODEL_DIR\
#    --tpu_name=$TPU_NAME\
#    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet'),
#
#    'densenet_bfloat16': ('densenet_bfloat16', 'python densenet_imagenet.py'\
#    + ' --alsologtostderr\
#    --steps_per_checkpoint=100\
#    --num_shards=8\
#    --mode=train\
#    --train_batch_size=$BATCH_SIZE\
#    --train_steps=$TRAIN_STEPS\
#    --iterations_per_loop=$ITERATIONS\
#    --model_dir=$MODEL_DIR\
#    --tpu_name=$TPU_NAME\
#    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet'),

#    'densenet_fake': ('densenet_fake', 'python densenet_imagenet.py'\
#    + ' --alsologtostderr\
#    --steps_per_checkpoint=100\
#    --num_shards=8\
#    --mode=train\
#    --train_batch_size=$BATCH_SIZE\
#    --train_steps=$TRAIN_STEPS\
#    --iterations_per_loop=$ITERATIONS\
#    --model_dir=$MODEL_DIR\
#    --tpu_name=$TPU_NAME\
#    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet'),

    'densenet_bfloat16_fake': ('densenet_bfloat16_fake', 'python densenet_imagenet.py'\
    + ' --alsologtostderr\
    --steps_per_checkpoint=100\
    --num_shards=8\
    --mode=train\
    --train_batch_size=$BATCH_SIZE\
    --train_steps=$TRAIN_STEPS\
    --iterations_per_loop=$ITERATIONS\
    --model_dir=$MODEL_DIR\
    --tpu_name=$TPU_NAME\
    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet'),

#    'mobilenet': ('mobilenet', 'python mobilenet.py' \
#    + ' --alsologtostderr\
#    --num_shards=8\
#    --mode=train\
#    --use_data=real\
#    --train_batch_size=$BATCH_SIZE\
#    --train_steps=$TRAIN_STEPS\
#    --iterations=$ITERATIONS\
#    --save_checkpoints_secs=10\
#    --model_dir=$MODEL_DIR\
#    --tpu_name=$TPU_NAME\
#    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet'),

#    'mobilenet_fake': ('mobilenet', 'python mobilenet.py' \
#    + ' --alsologtostderr\
#    --num_shards=8\
#    --mode=train\
#    --use_data=fake\
#    --train_batch_size=$BATCH_SIZE\
#    --train_steps=$TRAIN_STEPS\
#    --iterations=$ITERATIONS\
#    --save_checkpoints_secs=10\
#    --model_dir=$MODEL_DIR\
#    --tpu_name=$TPU_NAME\
#    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet'),

#    'mobilenet_bfloat16':('mobilenet_bfloat16', 'python mobilenet.py' \
#    + ' --alsologtostderr\
#    --num_shards=8\
#    --mode=train\
#    --use_data=real\
#    --train_batch_size=$BATCH_SIZE\
#    --train_steps=$TRAIN_STEPS\
#    --iterations=$ITERATIONS\
#    --save_checkpoints_secs=10\
#    --model_dir=$MODEL_DIR\
#    --tpu_name=$TPU_NAME\
#    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet'),

    'mobilenet_bfloat16_fake':('mobilenet_bfloat16', 'python mobilenet.py' \
    + ' --alsologtostderr\
    --num_shards=8\
    --mode=train\
    --use_data=fake\
    --train_batch_size=$BATCH_SIZE\
    --train_steps=$TRAIN_STEPS\
    --iterations=$ITERATIONS\
    --save_checkpoints_secs=10\
    --model_dir=$MODEL_DIR\
    --tpu_name=$TPU_NAME\
    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet'),

#    'retinanet':('retinanet', 'python retinanet_main.py'\
#    + ' --train_batch_size=$BATCH_SIZE\
#    --training_file_pattern=gs://$GCS_BUCKET_NAME/coco/train-* \
#    --resnet_checkpoint=gs://cloud-tpu-artifacts/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603 \
#     --model_dir=$MODEL_DIR\
#    --iterations_per_loop=$ITERATIONS\
#    --train_steps=$TRAIN_STEPS\
#    --hparams=image_size=640 \
#    --tpu_name=$TPU_NAME\
#    --num_examples_per_epoch=6400 \
#    --num_epochs=1'),
    
#    'retinanet_fake':('retinanet_fake', 'python retinanet_main.py'\
#    + ' --train_batch_size=$BATCH_SIZE\
#    --training_file_pattern=gs://$GCS_BUCKET_NAME/coco/train-* \
#    --resnet_checkpoint=gs://cloud-tpu-artifacts/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603 \
#     --model_dir=$MODEL_DIR\
#    --iterations_per_loop=$ITERATIONS\
#    --train_steps=$TRAIN_STEPS\
#    --hparams=image_size=640 \
#    --tpu_name=$TPU_NAME\
#    --num_examples_per_epoch=6400 \
#    --num_epochs=1'),

#    'retinanet_bfloat16':('retinanet_bfloat16', 'python retinanet_main.py'\
#    + ' --train_batch_size=$BATCH_SIZE\
#    --training_file_pattern=gs://$GCS_BUCKET_NAME/coco/train-* \
#    --resnet_checkpoint=gs://cloud-tpu-artifacts/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603 \
#     --model_dir=$MODEL_DIR\
#    --iterations_per_loop=$ITERATIONS\
#    --train_steps=$TRAIN_STEPS\
#    --hparams=image_size=640 \
#    --num_examples_per_epoch=6400 \
#    --tpu_name=$TPU_NAME\
#    --num_epochs=1'),
    
    'retinanet_bfloat16_fake':('retinanet_bfloat16_fake', 'python retinanet_main.py'\
    + ' --train_batch_size=$BATCH_SIZE\
    --training_file_pattern=gs://$GCS_BUCKET_NAME/coco/train-* \
    --resnet_checkpoint=gs://cloud-tpu-artifacts/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603 \
     --model_dir=$MODEL_DIR\
    --iterations_per_loop=$ITERATIONS\
    --train_steps=$TRAIN_STEPS\
    --hparams=image_size=640 \
    --num_examples_per_epoch=6400 \
    --tpu_name=$TPU_NAME\
    --num_epochs=1'),

#    'squeezenet':('squeezenet','python squeezenet_main.py' \
#    + ' --alsologtostderr\
#    --num_shards=8\
#    --num_evals=0\
#    --batch_size=$BATCH_SIZE\
#    --train_steps=$TRAIN_STEPS\
#    --iterations=$ITERATIONS\
#    --save_checkpoints_secs=10\
#    --model_dir=$MODEL_DIR\
#    --tpu_name=$TPU_NAME\
#    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet'),
#
#    'squeezenet_fake':('squeezenet_fake','python squeezenet_main.py' \
#    + ' --alsologtostderr\
#    --num_shards=8\
#    --num_evals=0\
#    --batch_size=$BATCH_SIZE\
#    --train_steps=$TRAIN_STEPS\
#    --iterations=$ITERATIONS\
#    --save_checkpoints_secs=10\
#    --model_dir=$MODEL_DIR\
#    --tpu_name=$TPU_NAME\
#    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet'),

#    'squeezenet_bfloat16':('squeezenet_bfloat16','python squeezenet_main.py' \
#    + ' --alsologtostderr\
#    --num_shards=8\
#    --num_evals=0\
#    --batch_size=$BATCH_SIZE\
#    --train_steps=$TRAIN_STEPS\
#    --iterations=$ITERATIONS\
#    --save_checkpoints_secs=10\
#    --model_dir=$MODEL_DIR\
#    --tpu_name=$TPU_NAME\
#    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet'),

    'squeezenet_bfloat16_fake':('squeezenet_bfloat16_fake','python squeezenet_main.py' \
    + ' --alsologtostderr\
    --num_shards=8\
    --num_evals=0\
    --batch_size=$BATCH_SIZE\
    --train_steps=$TRAIN_STEPS\
    --iterations=$ITERATIONS\
    --save_checkpoints_secs=10\
    --model_dir=$MODEL_DIR\
    --tpu_name=$TPU_NAME\
    --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet'),

#    'transformer':('',
#    '/home/wangyu/.local/bin/t2t-trainer \
#    --model=transformer \
#    --hparams_set=transformer_tpu \
#    --problem=translate_ende_wmt32k_packed \
#    --train_steps=500 \
#    --hparams=batch_size=$BATCH_SIZE,weight_dtype=float32 \
#    --eval_steps=1 \
#    --data_dir=gs://tpubenchmarking/transformer/data \
#    --output_dir=$MODEL_DIR \
#    --cloud_tpu_name=$TPU_NAME'
#    ),

    'transformer_bfloat16':('',
    '/home/wangyu/.local/bin/t2t-trainer \
    --model=transformer \
    --hparams_set=transformer_tpu \
    --problem=translate_ende_wmt32k_packed \
    --train_steps=$TRAIN_STEPS \
    --hparams=batch_size=$BATCH_SIZE \
    --eval_steps=1 \
    --data_dir=gs://tpubenchmarking/transformer/data \
    --output_dir=$MODEL_DIR \
    --cloud_tpu_name=$TPU_NAME'
    ),
    

    # for tf version < 1.11, t2t==1.6.5
#    'transformer_bfloat16':('',
#    '/home/wangyu/.local/bin/t2t-trainer \
#    --model=transformer \
#    --hparams_set=transformer_tpu \
#    --problem=translate_ende_wmt32k_packed \
#    --train_steps=$TRAIN_STEPS \
#    --hparams=batch_size=$BATCH_SIZE \
#    --eval_steps=1 \
#    --data_dir=gs://tpubenchmarking/transformer/data_1.8 \
#    --output_dir=$MODEL_DIR \
#    --master=grpc://10.240.1.146:8470 \
#    --tpu_num_shards=8 \
#    --use_tpu=True \
#    --zone=us-central1-f'
    ),
}

configs = {
  'resnet':(1024, 1000, 5000),
  'densenet':(1024, 1000, 5000),
  'mobilenet':(1024, 1000, 5000),
  'squeezenet':(1024, 1000, 5000),
  'retinanet':(64, 1000, 5000),
  'transformer':(4096, 100, 500),
}

def get_config(wl, configs):
  for k,v in configs.iteritems():
    if k in wl:
      return v
  return None

for name, (directory, cmd) in cmds.iteritems():
    (batch_size, iterations, train_steps) = get_config(name, configs)

    os.system('gsutil rm -r ' + tmp_dir)
    file_name = name + '-batchsize_' + str(batch_size) + '-iteration_' + str(iterations) + '-trainsteps_' + str(train_steps)

    os.system('grep \"global_step/sec\" ' + os.path.join(out_path, file_name + '.err') + ' > tmp')
    if not os.stat('tmp').st_size == 0:
        continue

    os.chdir(os.path.join(model_path, directory))
    if not 'BATCH_SIZE' in cmd:
      print(name, '\'s cmd does not have BATCH_SIZE.')
      continue
    if not 'MODEL_DIR' in cmd:
      print(name, '\'s cmd does not have MODEL_DIR.')
      continue
    if not 'TRAIN_STEPS' in cmd:
      print(name, '\'s cmd does not have TRAIN_STEPS.')
      continue
        
    cmd = cmd.replace('$GCS_BUCKET_NAME', GCS_BUCKET_NAME)
    cmd = cmd.replace('$BATCH_SIZE', str(batch_size))
    cmd = cmd.replace('$ITERATIONS', str(iterations))
    cmd = cmd.replace('$TRAIN_STEPS', str(train_steps))
    cmd = cmd.replace('$MODEL_DIR', tmp_dir)
    cmd = cmd.replace('$TPU_NAME', os.uname()[1])
    cmd += ' --use_tpu=True --zone=us-central1-f'
    cmd = " ".join(cmd.split())

    print(name, os.path.join(out_path, file_name + '.err'))
    print(cmd)
    outfile = open(os.path.join(out_path, file_name + '.out'), 'w')
    errfile = open(os.path.join(out_path, file_name + '.err'), 'w')
    p = subprocess.Popen(cmd.split(' '), stdout=outfile, stderr=errfile)
    p.wait()
