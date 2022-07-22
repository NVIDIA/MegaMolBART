import grpc
import torch
import logging
from concurrent import futures

from hydra import compose, initialize
from nemo_chem.models.megamolbart import NeMoMegaMolBARTWrapper
import megamolbart_pb2_grpc
from megamolbart_pb2 import OutputSpec
logger = logging.getLogger(__name__)


class InferenceService(megamolbart_pb2_grpc.GenerativeSampler):

    def __init__(self):
        if not hasattr(self, '_inferer'):
            with initialize(config_path="../../../../examples/chem/conf"):
                inf_cfg = compose(config_name="infer")
                self._inferer = NeMoMegaMolBARTWrapper(model_cfg=inf_cfg)


    def SmilesToEmbedding(self, spec, context):
        embeddings = self._inferer.smis_to_embedding(spec.smis)
        output = OutputSpec(embeddings=embeddings.flatten().tolist(),
                            dim=embeddings.shape)
        return output

    def SmilesToHidden(self, spec, context):
        hidden_states, pad_masks = self._inferer.smis_to_hidden(spec.smis)
        output = OutputSpec(hidden_states=hidden_states.flatten().tolist(),
                            dim=hidden_states.shape,
                            masks=pad_masks.flatten().tolist())
        return output

    def HiddenToSmis(self, spec, context):

        pad_mask = torch.BoolTensor(list(spec.masks))
        pad_mask = torch.reshape(pad_mask, tuple(spec.dim[:2])).cuda()

        hidden_states = torch.FloatTensor(list(spec.hidden_states))
        hidden_states = torch.reshape(hidden_states, tuple(spec.dim)).cuda()

        smis = self._inferer.hidden_to_smis(hidden_states,
                                            pad_mask)
        output = OutputSpec(smis=smis)
        return output

def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    megamolbart_pb2_grpc.add_GenerativeSamplerServicer_to_server(
        InferenceService(),
        server)
    server.add_insecure_port(f'[::]:{50051}')
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    main()