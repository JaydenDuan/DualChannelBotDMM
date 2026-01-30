import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.attention import StructuralAttention, TemporalAttention
from .layers.decoupler import ContentDecoupler, StructureDecoupler
from .layers.constraints import OrthogonalConstraint


class BotDMM(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, amr_size=768, num_prop_size=13,
                 llm_features_size=11, embedding_dimension=64, feature_dim=32,
                 num_temporal_steps=5, dropout=0.2, temperature=0.07, alpha=0.5,
                 num_heads=8, num_classes=3):
        super(BotDMM, self).__init__()
        self.num_temporal_steps = num_temporal_steps
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.num_classes = num_classes

        self.des_encoder = nn.Linear(des_size, feature_dim)
        self.tweet_encoder = nn.Linear(tweet_size, feature_dim)
        self.amr_encoder = nn.Linear(amr_size, feature_dim)
        self.num_prop_encoder = nn.Linear(num_prop_size, feature_dim)
        self.llm_features_encoder = nn.Linear(llm_features_size, feature_dim)

        self.content_decoupler = ContentDecoupler(5 * feature_dim, embedding_dimension, dropout)
        self.structure_decoupler = StructureDecoupler(5 * feature_dim, embedding_dimension, dropout)
        self.orthogonal_regularizer = OrthogonalConstraint(alpha=alpha)

        self.structure_projector = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.ReLU(),
            nn.Linear(embedding_dimension, embedding_dimension)
        )
        self.content_projector = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.ReLU(),
            nn.Linear(embedding_dimension, embedding_dimension)
        )
        self.class_projector = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LayerNorm(embedding_dimension),
            nn.ReLU(),
            nn.Linear(embedding_dimension, embedding_dimension)
        )

        self.tweet_lstm = nn.LSTM(
            input_size=tweet_size, hidden_size=feature_dim, num_layers=1,
            batch_first=True, bidirectional=False,
            dropout=dropout if num_temporal_steps > 1 else 0
        )
        self.amr_lstm = nn.LSTM(
            input_size=amr_size, hidden_size=feature_dim, num_layers=1,
            batch_first=True, bidirectional=False,
            dropout=dropout if num_temporal_steps > 1 else 0
        )

        self.feature_integrator = nn.Sequential(
            nn.Linear(embedding_dimension * 2, embedding_dimension),
            nn.LayerNorm(embedding_dimension),
            nn.ReLU()
        )
        self.temporal_gate = nn.Linear(embedding_dimension * 2, embedding_dimension)
        self.temporal_transform = nn.Sequential(
            nn.Linear(embedding_dimension * 2, embedding_dimension),
            nn.LayerNorm(embedding_dimension),
            nn.ReLU()
        )
        self.structural_attn_layers = nn.ModuleList([
            StructuralAttention(embedding_dimension, embedding_dimension, dropout)
            for _ in range(num_temporal_steps)
        ])
        self.temporal_pooling = TemporalAttention(embedding_dimension, dropout)

        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension // 2),
            nn.LayerNorm(embedding_dimension // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dimension // 2, num_classes)
        )

    def forward(self, des, tweets, amrs, num_prop, llm_features, edge_indices_list, labels=None):
        des_feat = self.des_encoder(des)
        num_feat = self.num_prop_encoder(num_prop)
        llm_feat = self.llm_features_encoder(llm_features)

        tweets_tensor = torch.stack(tweets, dim=1)
        tweet_lstm_out, _ = self.tweet_lstm(tweets_tensor)
        amrs_tensor = torch.stack(amrs, dim=1)
        amr_lstm_out, _ = self.amr_lstm(amrs_tensor)

        temporal_embeddings = []
        content_seq = []
        structure_seq = []
        steps_to_use = min(self.num_temporal_steps, len(tweets))

        for step in range(steps_to_use):
            tweet_feat = tweet_lstm_out[:, step, :]
            amr_feat = amr_lstm_out[:, step, :]
            combined_features = torch.cat([des_feat, tweet_feat, amr_feat, num_feat, llm_feat], dim=1)

            content_features = self.content_decoupler(combined_features)
            edge_index = edge_indices_list[step]

            if edge_index.size(1) > 0:
                max_idx = edge_index.max().item()
                if max_idx >= combined_features.size(0):
                    valid_mask = (edge_index[0] < combined_features.size(0)) & (edge_index[1] < combined_features.size(0))
                    edge_index = edge_index[:, valid_mask]

            structure_features = self.structure_decoupler(combined_features, edge_index)
            structure_features_norm, content_features_norm = self.orthogonal_regularizer(structure_features, content_features)

            structure_proj = F.normalize(self.structure_projector(structure_features), dim=1)
            content_proj = F.normalize(self.content_projector(content_features), dim=1)

            node_feat = torch.cat([structure_features_norm, content_features_norm], dim=1)
            node_feat = self.feature_integrator(node_feat)

            temporal_embeddings.append(node_feat)
            content_seq.append(content_proj)
            structure_seq.append(structure_proj)

        if len(temporal_embeddings) < self.num_temporal_steps:
            last_embedding = temporal_embeddings[-1]
            for _ in range(self.num_temporal_steps - len(temporal_embeddings)):
                temporal_embeddings.append(last_embedding)

        temporal_sequence = torch.stack(temporal_embeddings, dim=1)
        final_embedding = self.temporal_pooling(temporal_sequence)

        if labels is not None:
            proj_embedding = self.class_projector(final_embedding)
            class_contrastive_loss = self.class_contrastive_loss(proj_embedding, labels)
        else:
            class_contrastive_loss = torch.tensor(0.0, device=final_embedding.device)

        feature_contrastive_loss = self.feature_contrastive_loss(structure_proj, content_proj)
        logits = self.output_layer(final_embedding)

        return {
            'logits': logits,
            'final_embedding': final_embedding,
            'structure_features': structure_features,
            'content_features': content_features,
            'structure_proj': structure_proj,
            'content_proj': content_proj,
            'feature_contrastive_loss': feature_contrastive_loss,
            'class_contrastive_loss': class_contrastive_loss,
            'structure_seq': temporal_embeddings,
            'content_seq': content_seq,
        }

    def feature_contrastive_loss(self, structure_proj, content_proj):
        batch_size = structure_proj.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=structure_proj.device)

        similarity_matrix = torch.matmul(structure_proj, content_proj.T) / self.temperature
        similarity_matrix = torch.clamp(similarity_matrix, -50.0, 50.0)
        positives = torch.diag(similarity_matrix)
        mask = (~torch.eye(batch_size, device=structure_proj.device).bool()).float()
        exp_logits = torch.exp(similarity_matrix) * mask
        log_prob = positives - torch.log(exp_logits.sum(1) + 1e-8)
        loss = -log_prob.mean()

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=structure_proj.device)
        return loss

    def class_contrastive_loss(self, embeddings, labels):
        batch_size = embeddings.shape[0]
        if batch_size < 4:
            return torch.tensor(0.0, device=embeddings.device)

        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings_normalized, embeddings_normalized.T)
        similarity_matrix = torch.clamp(similarity_matrix / self.temperature, -50.0, 50.0)

        labels_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        mask_self = torch.eye(batch_size, device=embeddings.device)
        labels_matrix = labels_matrix * (1 - mask_self)

        has_positives = labels_matrix.sum(dim=1) > 0
        if not has_positives.any():
            return torch.tensor(0.0, device=embeddings.device)

        valid_indices = torch.where(has_positives)[0]
        exp_sim = torch.exp(similarity_matrix)
        pos_sim = torch.zeros(batch_size, device=embeddings.device)
        denom = torch.zeros(batch_size, device=embeddings.device)

        for i in valid_indices:
            pos_mask = labels_matrix[i]
            neg_mask = (1 - labels_matrix[i]) * (1 - mask_self[i])
            pos_sim[i] = (exp_sim[i] * pos_mask).sum()
            denom[i] = (exp_sim[i] * (pos_mask + neg_mask)).sum()

        valid_pos_sim = pos_sim[valid_indices]
        valid_denom = denom[valid_indices] + 1e-8
        loss = -torch.log(valid_pos_sim / valid_denom + 1e-8).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=embeddings.device)
        return loss
