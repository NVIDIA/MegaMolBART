import grpc
import torch
import logging
from megamolbart_pb2_grpc import GenerativeSamplerStub
from megamolbart_pb2 import InputSpec

log = logging.getLogger(__name__)


class InferenceWrapper():

    def __init__(self):
        channel = grpc.insecure_channel('localhost:50051')
        self.stub = GenerativeSamplerStub(channel)

    def smis_to_embedding(self, smis):
        spec = InputSpec(smis=smis)
        resp = self.stub.SmilesToEmbedding(spec)

        embeddings = torch.FloatTensor(list(resp.embeddings))
        embeddings = torch.reshape(embeddings, tuple(resp.dim)).cuda()

        return embeddings

    def smis_to_hidden(self, smis):
        spec = InputSpec(smis=smis)
        resp = self.stub.SmilesToHidden(spec)

        hidden_states = torch.FloatTensor(list(resp.hidden_states))
        hidden_states = torch.reshape(hidden_states, tuple(resp.dim)).cuda()
        masks = torch.BoolTensor(list(resp.masks))
        masks = torch.reshape(masks, tuple(resp.dim[:2])).cuda()

        return hidden_states, masks

    def hidden_to_smis(self, hidden_states, masks):
        dim = hidden_states.shape
        spec = InputSpec(hidden_states=hidden_states.flatten().tolist(),
                         dim=dim,
                         masks=masks.flatten().tolist())

        resp = self.stub.HiddenToSmis(spec)
        return resp.smis

