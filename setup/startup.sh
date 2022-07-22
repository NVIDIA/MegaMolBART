python3 -m nemo_chem.models.megamolbart.grpc.service &
jupyter lab \
    --no-browser \
    --port=8888 \
    --ip=0.0.0.0 \
    --allow-root \
    --notebook-dir=/opt/nvidia/nemo_chem/examples/chem/nbs \
    --NotebookApp.password='' \
    --NotebookApp.token='' \
    --NotebookApp.password_required=False