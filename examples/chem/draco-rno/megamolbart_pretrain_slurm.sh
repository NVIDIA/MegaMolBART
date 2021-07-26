#!/bin/bash
#SBATCH -A ent_aiapps_nlp
#SBATCH -p batch                                 # batch / batch_short / backfill
#SBATCH -N 2                                     # number of nodes
#SBATCH -t 8:00:00                               # wall time  (8 for batch, backfill, 2 for batch_short)
#SBATCH -J "nmt_megatron_3.9B"          # job name (<< CHANGE ! >>)
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --gpus-per-node=16      # n gpus per machine <required>
#SBATCH --ntasks-per-node=16    # n tasks per machine (one task per gpu) <required>
#SBATCH --overcommit            # Needed for pytorch
#SBATCH --gres=gpfs:circe       # Needed for Circe-Draco <required>
set -x
SLURM_ACCOUNT_DIR='ent_aiapps'  # <Make sure you dont override SLURM_ACCOUNT!>
USERID='eharper'
# << CHANGE THIS >>
CONTAINER="gitlab-master.nvidia.com/eharper/nemo_containers:nmt_megatron"

# Hyperparams
TOKENS_IN_BATCH=1000
LEARNING_RATE=1e-5
STEPS=150000
WARMUP_STEPS=15000

# Directories for manifests, data, etc.
# << CHANGE THIS >>
EXPNAME="megatron_en_de_3.9B" # exp_manager and wandb
PROJECT="nemo_nmt_enc_dec" # exp_manager and wandb

DATA="/gpfs/fs1/projects/ent_aiapps/users/eharper/data/68792"
PREPROC_DATA="/gpfs/fs1/projects/ent_aiapps/users/eharper/data/preproc_${EXPNAME}_tokens_${TOKENS_IN_BATCH}"
RESULTS_DIR="/gpfs/fs1/projects/${SLURM_ACCOUNT_DIR}/users/${USERID}/results/${EXPNAME}_tokens_${TOKENS_IN_BATCH}_nodes_${SLURM_JOB_NUM_NODES}"
CODE="/gpfs/fs1/projects/${SLURM_ACCOUNT_DIR}/users/${USERID}/repos/NeMo"
CHECKPOINT_FILE="/gpfs/fs1/projects/ent_aiapps/users/eharper/megatron/3.9b_bert_no_rng"

mkdir -p ${RESULTS_DIR}
mkdir -p ${PREPROC_DATA}

WANDB="c88d7d4769d01a5d43c5369f7f6759c2c02a78f2"
MOUNTS="--container-mounts=$CODE:/code,$RESULTS_DIR:/results,$DATA:/data,$PREPROC_DATA:/preproc_data,$CHECKPOINT_FILE:/checkpoint_file"

# Necessary Exports
export HYDRA_FULL_ERROR=1
# Make results dir
mkdir -p ${RESULTS_DIR}

OUTFILE="${RESULTS_DIR}/slurm-%j-%n.out"
ERRFILE="${RESULTS_DIR}/error-%j-%n.out"


read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& wandb login ${WANDB} \
&& echo "Starting training" \
&& CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 python /code/examples/nlp/machine_translation/enc_dec_nmt.py \
	--config-path=conf \
	--config-name=megatron \
	do_training=true \
	trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
	trainer.gpus=${SLURM_GPUS_PER_NODE} \
	~trainer.max_epochs \
	+trainer.max_steps=${STEPS} \
	+trainer.val_check_interval=1000 \
	+trainer.accumulate_grad_batches=1 \
	model.src_language=en \
	model.tgt_language=de \
	model.beam_size=3 \
	model.max_generation_delta=3 \
	model.label_smoothing=0.1 \
	model.preproc_out_dir=/preproc_data \
	model.encoder.checkpoint_file=/checkpoint_file \
	model.encoder.hidden_size=2560 \
	model.encoder.num_attention_heads=40 \
	model.encoder.num_layers=48 \
	model.encoder.max_position_embeddings=512 \
	model.decoder_tokenizer.library=yttm \
	model.decoder_tokenizer.vocab_size=32000 \
	model.decoder.library=nemo \
	model.decoder.pre_ln=True \
	model.decoder.num_layers=2 \
	model.decoder.hidden_size=3072 \
	model.decoder.inner_size=3072 \
	model.decoder.num_attention_heads=32 \
	model.decoder.ffn_dropout=0.1 \
	model.train_ds.use_tarred_dataset=true \
	model.train_ds.shard_strategy=replicate \
	model.train_ds.src_file_name=/data/train.clean.en.shuffled \
	model.train_ds.tgt_file_name=/data/train.clean.de.shuffled \
	model.train_ds.tokens_in_batch=${TOKENS_IN_BATCH} \
	model.validation_ds.src_file_name=[/data/wmt14-en-de.src,/data/wmt13-en-de.src] \
	model.validation_ds.tgt_file_name=[/data/wmt14-en-de.ref,/data/wmt13-en-de.ref] \
	~model.test_ds \
	model.optim.lr=$LEARNING_RATE  \
	model.optim.sched.name=WarmupAnnealing \
	+model.optim.sched.warmup_steps=$WARMUP_STEPS \
	~model.optim.sched.warmup_ratio \
	+exp_manager.create_wandb_logger=True \
	+exp_manager.wandb_logger_kwargs.name=${EXPNAME}_tokens_${TOKENS_IN_BATCH}_nodes_${SLURM_JOB_NUM_NODES} \
	+exp_manager.wandb_logger_kwargs.project=$PROJECT \
	+exp_manager.explicit_log_dir=/results \
	+exp_manager.resume_if_exists=True \
	+exp_manager.resume_ignore_no_checkpoint=True \
	+exp_manager.create_checkpoint_callback=True \
	+exp_manager.checkpoint_callback_params.monitor=val_sacreBLEU \
	+exp_manager.checkpoint_callback_params.save_top_k=1 \
	+exp_manager.checkpoint_callback_params.mode=max
EOF

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
set +x
