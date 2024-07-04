import torch
import torch.nn as nn
from einops import rearrange


class Prototype_Prompt_Encoder(nn.Module):
    def __init__(
        self,
        feat_dim=256,
        hidden_dim_dense=128,
        hidden_dim_sparse=128,
        size=64,
        num_tokens=8,
    ):
        super(Prototype_Prompt_Encoder, self).__init__()
        self.dense_fc_1 = nn.Conv2d(feat_dim, hidden_dim_dense, 1)
        self.dense_fc_2 = nn.Conv2d(hidden_dim_dense, feat_dim, 1)

        self.relu = nn.ReLU()

        self.sparse_fc_1 = nn.Conv1d(size * size, hidden_dim_sparse, 1)
        self.sparse_fc_2 = nn.Conv1d(hidden_dim_sparse, num_tokens, 1)

        pn_cls_embeddings = [
            nn.Embedding(num_tokens, feat_dim)
            for _ in range(2)  # One for positive and one for negative
        ]
        self.pn_cls_embeddings = nn.ModuleList(pn_cls_embeddings)

    def forward(self, feat, prototypes, cls_ids):
        # Ensuring prototypes are on the same device as feat
        cls_prompts = prototypes.unsqueeze(-1).to(feat.device)
        cls_prompts = torch.stack([cls_prompts for _ in range(feat.size(0))], dim=0)

        # Debugging device assignment
        print(f"Device of feat: {feat.device}")
        print(f"Device of cls_prompts: {cls_prompts.device}")

        # Expanding feat to match cls_prompts dimensions and ensuring device consistency
        feat_expanded = torch.stack(
            [feat for _ in range(cls_prompts.size(1))], dim=1
        ).to(feat.device)

        # Compute similarity matrix
        sim = torch.matmul(feat_expanded, cls_prompts)

        # Compute class-activated feature
        feat = feat_expanded + feat_expanded * sim

        # Process for dense embeddings
        feat_dense = feat.clone()
        one_hot = torch.nn.functional.one_hot(cls_ids, 7).to(feat.device)

        # Debugging shapes before reshaping
        print(
            f"feat_dense shape before reshape: {feat_dense.shape}"
        )  # [16, 16, 4096, 256]
        print(f"one_hot shape: {one_hot.shape}")  # [16, 7]

        # Ensuring that one_hot matches feat_dense shape for indexing
        feat_dense = rearrange(feat_dense, "b num_cls hw c -> b hw num_cls c")
        one_hot = rearrange(one_hot, "b n -> b 1 n").bool()

        # Debugging shapes after reshaping
        print(
            f"feat_dense shape after reshape: {feat_dense.shape}"
        )  # [16, 4096, 16, 256]
        print(f"one_hot shape after reshape: {one_hot.shape}")  # [16, 1, 7]

        # Verify content of one_hot tensor for debugging
        print(f"one_hot tensor content: {one_hot}")

        # Select features for the given classes
        try:
            selected_feat_dense = torch.masked_select(
                feat_dense, one_hot.unsqueeze(-1)
            ).view(feat_dense.size(0), -1, 64, 64, 256)
        except RuntimeError as e:
            print(f"Error during masked_select: {e}")
            print(f"feat_dense shape during error: {feat_dense.shape}")
            print(f"one_hot shape during error: {one_hot.unsqueeze(-1).shape}")
            raise e

        # Debugging shape after selection and rearrange
        print(f"selected_feat_dense shape: {selected_feat_dense.shape}")

        dense_embeddings = self.dense_fc_2(
            self.relu(self.dense_fc_1(selected_feat_dense))
        )

        # Process for sparse embeddings
        feat_sparse = rearrange(feat, "b num_cls hw c -> (b num_cls) hw c")
        sparse_embeddings = self.sparse_fc_2(self.relu(self.sparse_fc_1(feat_sparse)))
        sparse_embeddings = rearrange(
            sparse_embeddings, "(b num_cls) n c -> b num_cls n c", num_cls=7
        )

        pos_embed = self.pn_cls_embeddings[1].weight.unsqueeze(0).unsqueeze(
            0
        ) * one_hot.unsqueeze(-1).unsqueeze(-1)
        neg_embed = self.pn_cls_embeddings[0].weight.unsqueeze(0).unsqueeze(0) * (
            1 - one_hot
        ).unsqueeze(-1).unsqueeze(-1)

        sparse_embeddings = sparse_embeddings + pos_embed.detach() + neg_embed.detach()

        sparse_embeddings = rearrange(
            sparse_embeddings, "b num_cls n c -> b (num_cls n) c"
        )

        return dense_embeddings, sparse_embeddings


class Learnable_Prototypes(nn.Module):
    def __init__(self, num_classes=7, feat_dim=256, clip_embeddings=None):
        super(Learnable_Prototypes, self).__init__()
        self.class_embeddings = nn.Embedding(num_classes, feat_dim)
        # initialize the class embeddings with the clip embeddings
        if clip_embeddings is not None:
            print("Initializing prototypes with CLIP embeddings.")
            self.class_embeddings.weight.data.copy_(clip_embeddings)
            self.class_embeddings.weight.requires_grad = (
                False  # Ensuring they are not trainable
            )

    def forward(self):
        return self.class_embeddings.weight
