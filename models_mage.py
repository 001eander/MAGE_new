from functools import partial
import math

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, DropPath, Mlp

from util.pos_embed import get_2d_sincos_pos_embed

from taming.models.vqgan import VQModel
from omegaconf import OmegaConf
import numpy as np
import scipy.stats as stats


def unknown_count_at_step(step, num_iter=12, n_tokens=256):
    """How many tokens are still masked at the start of a gen_image step."""
    unknown = int(n_tokens)
    for s in range(int(step)):
        ratio = (s + 1) / float(num_iter)
        mask_ratio = math.cos(math.pi / 2.0 * ratio)
        mask_len = int(math.floor(n_tokens * mask_ratio))
        mask_len = max(1, min(unknown - 1, mask_len))
        unknown = mask_len
    return unknown


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        with torch.cuda.amp.autocast(enabled=False):
            attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale

        attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            _, attn = self.attn(self.norm1(x))
            return attn
        else:
            y, _ = self.attn(self.norm1(x))
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, vocab_size, hidden_size, max_position_embeddings, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

        torch.nn.init.normal_(self.word_embeddings.weight, std=.02)
        torch.nn.init.normal_(self.position_embeddings.weight, std=.02)

    def forward(
        self, input_ids
    ):
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        position_ids = self.position_ids[:, :seq_length]

        inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MlmLayer(nn.Module):

    def __init__(self, feat_emb_dim, word_emb_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(feat_emb_dim, word_emb_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(word_emb_dim)
        self.bias = nn.Parameter(torch.zeros(1, 1, vocab_size))

    def forward(self, x, word_embeddings):
        mlm_hidden = self.fc(x)
        mlm_hidden = self.gelu(mlm_hidden)
        mlm_hidden = self.ln(mlm_hidden)
        word_embeddings = word_embeddings.transpose(0, 1)
        logits = torch.matmul(mlm_hidden, word_embeddings)
        logits = logits + self.bias
        return logits


class MaskedGenerativeEncoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_ratio_min=0.5, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25,
                 label_smoothing=0.1, dropout=0.1, fixed_mask_ratio=None,
                 full_mask_prob=0.0, gen_schedule_prob=0.0, gen_num_iter=12,
                 decoder_mask_token=False, linear_head=False, use_cond_ids=False,
                 vqgan_ckpt_path='vqgan_jax_strongaug.ckpt'):
        super().__init__()

        # --------------------------------------------------------------------------
        # VQGAN specifics
        config = OmegaConf.load('config/vqgan.yaml').model
        self.vqgan = VQModel(ddconfig=config.params.ddconfig,
                             n_embed=config.params.n_embed,
                             embed_dim=config.params.embed_dim,
                             ckpt_path=vqgan_ckpt_path)
        for param in self.vqgan.parameters():
            param.requires_grad = False

        self.codebook_size = config.params.n_embed
        self.codebook_emb_dim = config.params.embed_dim
        vocab_size = self.codebook_size + 1000 + 1  # 1024 codebook size, 1000 classes, 1 for mask token.
        self.fake_class_label = self.codebook_size + 1100 - 1024
        self.mask_token_label = vocab_size - 1
        self.token_emb = BertEmbeddings(vocab_size=vocab_size,
                                        hidden_size=embed_dim,
                                        max_position_embeddings=256+1,
                                        dropout=dropout)

        # MAGE variant masking ratio
        self.mask_ratio_min = mask_ratio_min
        self.fixed_mask_ratio = fixed_mask_ratio
        self.full_mask_prob = full_mask_prob
        self.gen_schedule_prob = gen_schedule_prob
        self.gen_num_iter = gen_num_iter
        self.use_cond_ids = use_cond_ids
        self._cached_gt_indices = None
        self._cached_gt_by_id = None
        if fixed_mask_ratio is None:
            self.mask_ratio_generator = stats.truncnorm(
                (mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                loc=mask_ratio_mu, scale=mask_ratio_std)
        else:
            self.mask_ratio_generator = None

        # --------------------------------------------------------------------------
        # MAGE encoder specifics
        dropout_rate = dropout
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAGE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pad_with_cls_token = not decoder_mask_token
        self.linear_head = linear_head

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))  # learnable pos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MlmLayer
        self.mlm_layer = MlmLayer(feat_emb_dim=decoder_embed_dim, word_emb_dim=embed_dim, vocab_size=vocab_size)
        self.token_classifier = nn.Linear(decoder_embed_dim, self.codebook_size) if linear_head else None

        self.norm_pix_loss = norm_pix_loss

        self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        pos_std = 0.2 if not self.pad_with_cls_token else 0.02
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=pos_std)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _masks_with_counts(self, bsz, seq_len, num_dropped_tokens, num_masked_tokens, device):
        num_dropped_tokens = min(max(int(num_dropped_tokens), 0), seq_len)
        num_masked_tokens = min(max(int(num_masked_tokens), num_dropped_tokens), seq_len)
        if num_dropped_tokens == 0 and num_masked_tokens == 0:
            zeros = torch.zeros(bsz, seq_len, device=device)
            return zeros, zeros.clone()
        if num_dropped_tokens == 0 and num_masked_tokens == seq_len:
            return torch.zeros(bsz, seq_len, device=device), torch.ones(bsz, seq_len, device=device)

        while True:
            noise = torch.rand(bsz, seq_len, device=device)
            sorted_noise, _ = torch.sort(noise, dim=1)
            if num_dropped_tokens == 0:
                token_drop_mask = torch.zeros(bsz, seq_len, device=device)
            else:
                cutoff_drop = sorted_noise[:, num_dropped_tokens - 1:num_dropped_tokens]
                token_drop_mask = (noise <= cutoff_drop).float()
            if num_masked_tokens == seq_len:
                token_all_mask = torch.ones(bsz, seq_len, device=device)
            else:
                cutoff_mask = sorted_noise[:, num_masked_tokens - 1:num_masked_tokens]
                token_all_mask = (noise <= cutoff_mask).float()
            drop_ok = int(token_drop_mask.sum().item()) == bsz * num_dropped_tokens
            mask_ok = int(token_all_mask.sum().item()) == bsz * num_masked_tokens
            if drop_ok and mask_ok:
                break
            print("Rerandom the noise!")
        return token_drop_mask, token_all_mask

    def _sample_masks(self, bsz, seq_len, device):
        u = float(np.random.rand())
        full_p = float(self.full_mask_prob or 0.0)
        sched_p = float(self.gen_schedule_prob or 0.0)
        if u < full_p:
            return self._masks_with_counts(bsz, seq_len, 0, seq_len, device)
        if u < full_p + sched_p:
            step = int(np.random.randint(0, max(int(self.gen_num_iter), 1)))
            n_mask = unknown_count_at_step(step, int(self.gen_num_iter), seq_len)
            return self._masks_with_counts(bsz, seq_len, 0, n_mask, device)
        if self.fixed_mask_ratio is not None:
            mask_rate = float(self.fixed_mask_ratio)
        else:
            mask_rate = float(self.mask_ratio_generator.rvs(1)[0])
        num_dropped_tokens = int(np.ceil(seq_len * float(self.mask_ratio_min)))
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        return self._masks_with_counts(bsz, seq_len, num_dropped_tokens, num_masked_tokens, device)

    def _prepend_cls(self, token_indices, token_drop_mask, token_all_mask, cond_ids=None):
        device = token_indices.device
        bsz = token_indices.size(0)
        cls = torch.zeros(bsz, 1, device=device, dtype=token_indices.dtype)
        token_indices = torch.cat([cls, token_indices], dim=1)
        if self.use_cond_ids:
            if cond_ids is None:
                raise ValueError('use_cond_ids requires cond_ids')
            cond = cond_ids.to(device=device, dtype=token_indices.dtype).view(-1)
            if cond.numel() == 1 and bsz > 1:
                cond = cond.expand(bsz)
            token_indices[:, 0] = self.codebook_size + cond
        else:
            token_indices[:, 0] = self.fake_class_label
        pad = torch.zeros(bsz, 1, device=device, dtype=token_drop_mask.dtype)
        token_drop_mask = torch.cat([pad, token_drop_mask], dim=1)
        token_all_mask = torch.cat([pad, token_all_mask], dim=1)
        return token_indices.long(), token_drop_mask, token_all_mask

    def _run_encoder(self, token_indices, token_drop_mask):
        input_embeddings = self.token_emb(token_indices)
        bsz, _, emb_dim = input_embeddings.shape
        token_keep_mask = 1 - token_drop_mask
        x = input_embeddings[token_keep_mask.nonzero(as_tuple=True)].reshape(bsz, -1, emb_dim)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    @torch.no_grad()
    def vqgan_encode(self, imgs):
        z_q, _, token_tuple = self.vqgan.encode(imgs)
        _, _, token_indices = token_tuple
        return token_indices.reshape(z_q.size(0), -1).long()

    @torch.no_grad()
    def encode_to_indices(self, imgs, cond_ids=None):
        cache_map = getattr(self, '_cached_gt_by_id', None)
        if cache_map and cond_ids is not None:
            rows = []
            for i in cond_ids.view(-1).tolist():
                t = cache_map[int(i)]
                if t.device != imgs.device:
                    t = t.to(imgs.device)
                    cache_map[int(i)] = t
                rows.append(t)
            return torch.stack(rows, 0)
        cached = getattr(self, '_cached_gt_indices', None)
        if cached is not None:
            if cached.device != imgs.device:
                cached = cached.to(imgs.device)
                self._cached_gt_indices = cached
            if cached.size(0) != imgs.size(0):
                cached = cached[:1].expand(imgs.size(0), -1).contiguous()
            return cached
        return self.vqgan_encode(imgs)

    @torch.no_grad()
    def decode_indices(self, indices):
        bsz = indices.size(0)
        z_q = self.vqgan.quantize.get_codebook_entry(
            indices.long(), shape=(bsz, 16, 16, self.codebook_emb_dim))
        return self.vqgan.decode(z_q)

    @torch.no_grad()
    def predict_full_mask(self, imgs, cond_ids=None):
        """One-shot greedy tokens from an all-mask input (generation-aligned)."""
        gt_indices = self.encode_to_indices(imgs, cond_ids)
        bsz, seq_len = gt_indices.shape
        device = gt_indices.device
        token_indices = torch.full(
            (bsz, seq_len), self.mask_token_label, device=device, dtype=torch.long)
        token_drop_mask = torch.zeros(bsz, seq_len, device=device)
        token_all_mask = torch.ones(bsz, seq_len, device=device)
        token_indices, token_drop_mask, token_all_mask = self._prepend_cls(
            token_indices, token_drop_mask, token_all_mask, cond_ids=cond_ids)
        latent = self._run_encoder(token_indices, token_drop_mask)
        logits = self.forward_decoder(latent, token_drop_mask, token_all_mask)
        pred = logits[:, 1:, :self.codebook_size].argmax(dim=-1)
        return pred, gt_indices

    def forward_encoder(self, x, cond_ids=None):
        gt_indices = self.encode_to_indices(x, cond_ids)
        token_drop_mask, token_all_mask = self._sample_masks(
            gt_indices.size(0), gt_indices.size(1), x.device)
        token_indices = gt_indices.clone()
        token_indices[token_all_mask.bool()] = self.mask_token_label
        token_indices, token_drop_mask, token_all_mask = self._prepend_cls(
            token_indices, token_drop_mask, token_all_mask, cond_ids=cond_ids)
        latent = self._run_encoder(token_indices, token_drop_mask)
        return latent, gt_indices, token_drop_mask, token_all_mask

    def forward_decoder(self, x, token_drop_mask, token_all_mask):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        if self.pad_with_cls_token:
            mask_tokens = x[:, 0:1].repeat(1, token_all_mask.shape[1], 1)
        else:
            mask_tokens = self.mask_token.repeat(token_all_mask.shape[0], token_all_mask.shape[1], 1)

        # put undropped tokens into original sequence
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - token_drop_mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        # set undropped but masked positions with mask
        x_after_pad = torch.where(token_all_mask.unsqueeze(-1).bool(), mask_tokens, x_after_pad)

        # add pos embed
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        if self.linear_head:
            return self.token_classifier(x)

        word_embeddings = self.token_emb.word_embeddings.weight.data.detach()
        x = self.mlm_layer(x, word_embeddings)
        return x

    def forward_loss(self, gt_indices, logits, mask):
        bsz, seq_len = gt_indices.size()
        # logits and mask are with seq_len+1 but gt_indices is with seq_len
        logits_tok = logits[:, 1:, :self.codebook_size]
        loss = self.criterion(logits_tok.reshape(bsz * seq_len, -1), gt_indices.reshape(bsz * seq_len))
        loss = loss.reshape(bsz, seq_len)
        mask_tok = mask[:, 1:]
        denom = mask_tok.sum().clamp(min=1.0)
        loss = (loss * mask_tok).sum() / denom
        with torch.no_grad():
            pred = logits_tok.argmax(dim=-1)
            acc = ((pred == gt_indices).float() * mask_tok).sum() / denom
        return loss, acc

    def forward(self, imgs, cond_ids=None):
        latent, gt_indices, token_drop_mask, token_all_mask = self.forward_encoder(imgs, cond_ids)
        logits = self.forward_decoder(latent, token_drop_mask, token_all_mask)
        loss, acc = self.forward_loss(gt_indices, logits, token_all_mask)
        return loss, acc, token_all_mask


def mage_vit_base_patch16(**kwargs):
    model = MaskedGenerativeEncoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mage_vit_large_patch16(**kwargs):
    model = MaskedGenerativeEncoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
