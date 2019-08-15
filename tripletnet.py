import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class ConceptBranch(nn.Module):
    def __init__(self, out_dim, embedding_dim):
        super(ConceptBranch, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(embedding_dim, 32), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(32, out_dim), nn.Softmax())
                                 
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class CS_Tripletnet(nn.Module):
    def __init__(self, embeddingnet, num_concepts):
        super(CS_Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet
        self.num_concepts = num_concepts
        self.concept_branch = ConceptBranch(self.num_concepts, 64*3)

    def forward(self, x, y, z, c):
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            c: Integer indicating according to which notion of similarity images are compared"""
       
        general_x = self.embeddingnet.embeddingnet(x)
        general_y = self.embeddingnet.embeddingnet(y)
        general_z = self.embeddingnet.embeddingnet(z)
        # l2-normalize embeddings
        norm = torch.norm(general_x, p=2, dim=1) + 1e-10
        general_x = general_x / norm.expand_as(general_x)
        norm = torch.norm(general_y, p=2, dim=1) + 1e-10
        general_y = general_y / norm.expand_as(general_y)
        norm = torch.norm(general_z, p=2, dim=1) + 1e-10
        general_z = general_z / norm.expand_as(general_z)

        feat = torch.cat((general_x, general_y), 1)
        feat = torch.cat((feat, general_z), 1)
        weights_xy = self.concept_branch(feat)
        embedded_x = None
        embedded_y = None
        embedded_z = None
        mask_norm = None
        for idx in range(self.num_concepts):
            concept_idx = np.zeros((len(x),), dtype=int)
            concept_idx += idx
            concept_idx = torch.from_numpy(concept_idx)
            concept_idx = concept_idx.cuda()
            concept_idx = Variable(concept_idx)

            tmp_embedded_x, masknorm_norm_x, embed_norm_x, tot_embed_norm_x = self.embeddingnet(x, concept_idx)
            tmp_embedded_y, masknorm_norm_y, embed_norm_y, tot_embed_norm_y = self.embeddingnet(y, concept_idx)
            tmp_embedded_z, masknorm_norm_z, embed_norm_z, tot_embed_norm_z = self.embeddingnet(z, concept_idx)
 
            if mask_norm is None:
                mask_norm = masknorm_norm_x
            else:
                mask_norm += masknorm_norm_x

            weights = weights_xy[:, idx]
            weights = weights.unsqueeze(1)
            if embedded_x is None:
                embedded_x = tmp_embedded_x * weights.expand_as(tmp_embedded_x)
                embedded_y = tmp_embedded_y * weights.expand_as(tmp_embedded_y)
                embedded_z = tmp_embedded_z * weights.expand_as(tmp_embedded_z)
            else:
                embedded_x += tmp_embedded_x * weights.expand_as(tmp_embedded_x)
                embedded_y += tmp_embedded_y * weights.expand_as(tmp_embedded_y)
                embedded_z += tmp_embedded_z * weights.expand_as(tmp_embedded_z)

        mask_norm /= self.num_concepts
        embed_norm = (embed_norm_x + embed_norm_y + embed_norm_z) / 3
        mask_embed_norm = (tot_embed_norm_x + tot_embed_norm_y + tot_embed_norm_z) / 3
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, mask_norm, embed_norm, mask_embed_norm
