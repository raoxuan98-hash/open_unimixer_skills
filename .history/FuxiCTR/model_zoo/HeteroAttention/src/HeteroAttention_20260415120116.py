import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


class TransformerCTR(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="TransformerCTR",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 num_transformer_layers=2,
                 num_heads=2,
                 transformer_dropout=0.1,
                 ffn_dim=128,
                 use_cls_token=True,
                 use_pos_embedding=True,
                 output_mlp_hidden_units=[128, 64],
                 net_dropout=0.0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        """
        Args:
            feature_map: FeatureMap object from FuxiCTR.
            embedding_dim: Dimension of feature embeddings.
            num_transformer_layers: Number of Transformer encoder layers.
            num_heads: Number of attention heads.
            transformer_dropout: Dropout rate inside Transformer layers.
            ffn_dim: Hidden dimension of the Feed-Forward Network in Transformer.
            use_cls_token: If True, prepend a learnable CLS token and use its output
                           for final prediction (similar to BERT).
            use_pos_embedding: If True, add learnable positional embeddings.
            output_mlp_hidden_units: Hidden units of the final output MLP.
            net_dropout: Dropout rate of the output MLP.
            batch_norm: Whether to use batch normalization in the output MLP.
        """
        super(TransformerCTR, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs)

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.use_cls_token = use_cls_token
        self.use_pos_embedding = use_pos_embedding

        self.num_fields = feature_map.num_fields
        self.seq_len = self.num_fields + (1 if use_cls_token else 0)

        # CLS token: learnable parameter of shape [1, 1, embedding_dim]
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        # Positional Embedding: learnable parameter of shape [1, seq_len, embedding_dim]
        if self.use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, embedding_dim))

        # Transformer Encoder Layers (PyTorch native implementation)
        # batch_first=True means input shape is (batch, seq, feature)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=transformer_dropout,
            activation='relu',
            batch_first=True,
            norm_first=False  # Post-LayerNorm; set True for Pre-LayerNorm
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # Output MLP
        if self.use_cls_token:
            mlp_input_dim = embedding_dim
        else:
            mlp_input_dim = self.num_fields * embedding_dim

        self.output_mlp = MLP_Block(
            input_dim=mlp_input_dim,
            output_dim=1,
            hidden_units=output_mlp_hidden_units,
            hidden_activations='relu',
            output_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm
        )

        # FuxiCTR lifecycle methods
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Forward pass.

        Args:
            inputs: dict containing feature tensors and labels.

        Returns:
            dict: {"y_pred": tensor of shape (batch_size, 1)}
        """
        X = self.get_inputs(inputs)

        # 1. Feature Embedding: (batch_size, num_fields, embedding_dim)
        feature_emb = self.embedding_layer(X)

        # 2. Optional: prepend CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(feature_emb.size(0), -1, -1)
            x = torch.cat([cls_tokens, feature_emb], dim=1)  # (batch, seq_len, emb_dim)
        else:
            x = feature_emb

        # 3. Optional: add positional embeddings
        if self.use_pos_embedding:
            x = x + self.pos_embedding

        # 4. Transformer Encoder
        # For CTR data, there is typically no padding inside the feature sequence,
        # so we don't need src_key_padding_mask here.
        transformer_out = self.transformer_encoder(x)  # (batch, seq_len, emb_dim)

        # 5. Aggregation strategy
        if self.use_cls_token:
            # Use the output corresponding to the CLS token
            output = transformer_out[:, 0, :]  # (batch, emb_dim)
        else:
            # Flatten all feature-token outputs
            output = transformer_out.flatten(start_dim=1)  # (batch, num_fields * emb_dim)

        # 6. Final prediction
        y_pred = self.output_mlp(output)
        y_pred = self.output_activation(y_pred)

        return {"y_pred": y_pred}
