import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class ProbSparseAttention(nn.Module):
    """
    Atenção esparsa inspirada no Informer.
    Seleciona apenas os top-k queries mais relevantes.
    """
    def __init__(self, d_model, num_heads, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Projeções lineares para Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Calcula as top queries baseado na distribuição de probabilidade
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Calcular Top-k queries
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=K.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Encontrar Top_k queries com máxima variação
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top_values, M_top_indices = M.topk(n_top, sorted=False)
        
        # Calcular scores para as top queries
        # Gather the corresponding Q and K for top queries
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top_indices, :]  # (B, H, n_top, E)
        
        # Calcular scores completos para as top queries
        scores_top = torch.matmul(Q_reduce, K.transpose(-2, -1))  # (B, H, n_top, L_K)

        return scores_top, M_top_indices

    def _get_initial_context(self, V, L_Q):
        """
        Contexto inicial V
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            # Usar máscara causal
            assert(L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        """
        Atualizar contexto baseado nas queries top-k
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        
        attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
        attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
        
        return context_in, attns

    def forward(self, query, key, value):
        B, L, D = query.shape
        
        # Projetar Q, K, V
        Q = self.q_proj(query)  # (B, L, D)
        K = self.k_proj(key)    # (B, L, D)
        V = self.v_proj(value)  # (B, L, D)
        
        # Reshape para multi-head: (B, L, D) -> (B, L, H, head_dim) -> (B, H, L, head_dim)
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        L_Q = L_K = L
        
        # Usar atenção esparsa apenas se a sequência for longa o suficiente
        if L > 25:  # Threshold para usar atenção esparsa
            U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
            u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
            
            U_part = U_part if U_part < L_K else L_K
            u = u if u < L_Q else L_Q
            
            scores_top, index = self._prob_QK(Q, K, sample_k=U_part, n_top=u)

            # Adicionar scale factor
            scale = self.scale or 1./math.sqrt(self.head_dim)
            if scale is not None:
                scores_top = scores_top * scale
            
            # Obter contexto
            context = self._get_initial_context(V, L_Q)
            
            # Atualizar contexto
            context, attn = self._update_context(context, V, scores_top, index, L_Q)
        else:
            # Usar atenção padrão para sequências curtas
            scale = 1./math.sqrt(self.head_dim)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            
            if self.mask_flag:
                # Aplicar máscara causal
                mask = torch.triu(torch.ones(L, L, device=query.device), diagonal=1).bool()
                scores.masked_fill_(mask, -np.inf)
            
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            context = torch.matmul(attn, V)
        
        # Reshape de volta: (B, H, L, head_dim) -> (B, L, H, head_dim) -> (B, L, D)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        
        # Projeção final
        output = self.out_proj(context)
        
        return output, None  # Retornar None para compatibilidade


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self.mask = indicator.view(scores.shape).to(device)


class DistillingOperation(nn.Module):
    """
    Operação de destilação que reduz a dimensão temporal
    """
    def __init__(self, d_model, factor=2):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            stride=factor
        )
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ELU()

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # Aplicar normalização
        x = self.norm(x)
        
        # Convolução para reduzir dimensão temporal
        # Transpor para (batch, d_model, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.activation(x)
        # Voltar para (batch, new_seq_len, d_model)
        x = x.permute(0, 2, 1)
        
        return x


class InformerInspiredTransformer(nn.Module):
    """
    Transformer inspirado no Informer com atenção esparsa e destilação
    """
    def __init__(self, input_dims, output_dims, num_heads=8, num_layers=3, 
                 max_len=3000, dropout=0.1, factor=5, distil=True):
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.distil = distil
        
        # Ajustar num_heads para ser divisível por output_dims
        self.num_heads = self._adjust_num_heads(output_dims, num_heads)
        
        print(f"🔧 InformerInspiredTransformer: output_dims={output_dims}, num_heads={self.num_heads}")
        
        # Projeção inicial
        self.input_proj = nn.Linear(input_dims, output_dims)
        
        # Codificação posicional
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, output_dims) * 0.1)
        
        # Camadas do encoder
        self.encoder_layers = nn.ModuleList()
        self.distil_layers = nn.ModuleList() if distil else None
        
        for i in range(num_layers):
            # Camada de atenção
            encoder_layer = InformerEncoderLayer(
                d_model=output_dims,
                nhead=self.num_heads,
                dim_feedforward=output_dims * 4,
                dropout=dropout,
                factor=factor
            )
            self.encoder_layers.append(encoder_layer)
            
            # Camada de destilação (exceto na última camada)
            if distil and i < num_layers - 1:
                distil_layer = DistillingOperation(output_dims, factor=2)
                self.distil_layers.append(distil_layer)
        
        # Projeção final
        self.output_proj = nn.Linear(output_dims, output_dims)
        self.final_norm = nn.LayerNorm(output_dims)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _adjust_num_heads(self, embed_dim, desired_heads):
        """Ajusta número de cabeças para ser divisível"""
        divisors = [i for i in range(1, embed_dim + 1) if embed_dim % i == 0]
        return min(divisors, key=lambda x: abs(x - desired_heads))
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Projeção inicial
        x = self.input_proj(x)
        
        # Adicionar codificação posicional
        x = x + self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Passar pelas camadas encoder + destilação
        for i, encoder_layer in enumerate(self.encoder_layers):
            # Atenção
            x = encoder_layer(x)
            
            # Destilação (exceto na última camada)
            if self.distil and i < len(self.encoder_layers) - 1:
                if i < len(self.distil_layers):
                    x = self.distil_layers[i](x)
        
        # Projeção final
        x = self.output_proj(x)
        x = self.final_norm(x)
        
        return x


class InformerEncoderLayer(nn.Module):
    """
    Camada do encoder inspirada no Informer
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, factor=5):
        super().__init__()
        
        self.self_attn = ProbSparseAttention(
            d_model=d_model,
            num_heads=nhead,
            factor=factor, 
            attention_dropout=dropout
        )
        
        # Feed forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()
    
    def forward(self, src):
        # Self-attention
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


def create_informer_inspired_transformer(input_dims, output_dims, num_heads=8, num_layers=3, 
                                       max_len=3000, dropout=0.1, factor=5, distil=True):
    """
    Função factory para criar o InformerInspiredTransformer
    """
    return InformerInspiredTransformer(
        input_dims=input_dims,
        output_dims=output_dims,
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=max_len,
        dropout=dropout,
        factor=factor,
        distil=distil
    )