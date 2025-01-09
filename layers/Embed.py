import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)#在最后一个维度上进行pooling

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)#对pool之后的数据进行填充，保证形状一致（len-kneral）/stide+1
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    


class AutoConCI(nn.Module):  # AutoCon Channel Independence (CI) version for Multivariate
    def __init__(self, batch_size, seq_len, acf_values, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(AutoConCI, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.acf_values = torch.from_numpy(acf_values)
        self.seq_len = seq_len

    def local_contrastive_loss(self, features, labels):
        BC, T, D = features.shape

        # Compute local representation similarities
        local_features = features.clone()  # BC, T, D

        anchor_dot_contrast = torch.div(
            torch.bmm(local_features, local_features.transpose(1, 2)),  # (BC, T, D) X (BC, D, T) > (BC, T, T)
            self.temperature)
        # for numerical stability

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        local_logits = anchor_dot_contrast - logits_max.detach()  # subtract most large value

        # tile mask
        local_distmap = (labels.unsqueeze(1) - labels.unsqueeze(2)).abs()  # (BC, T, T) ,  (C, L)
        local_distmap = local_distmap.reshape(self.batch_size, -1, T*T).cpu()
        acf_values = self.acf_values.unsqueeze(0).repeat(self.batch_size,1, 1)
        local_distmap = torch.gather(acf_values, 2, local_distmap).float().to(features.get_device())
        local_distmap = local_distmap.reshape(-1, T, T)

        neg_mask = torch.scatter(
            torch.ones_like(local_distmap).to(features.get_device()),
            2,
            torch.arange(T).reshape(1, -1, 1).repeat(BC, 1, 1).to(features.get_device()),
            0
        )

        self_mask = (local_distmap == 1.0)
        pos_mask = local_autocorr_mask(local_distmap, self_mask) + (neg_mask * self_mask)

        exp_local_logits = torch.exp(local_logits) * neg_mask  # denominator

        log_local_prob = local_logits - torch.log(exp_local_logits.sum(2, keepdim=True))  # (B, T, T) > (B, T ,1)

        mean_log_local_prob_pos = (local_distmap * pos_mask * log_local_prob).sum(2) / pos_mask.sum(2)

        local_loss = - (self.temperature / self.base_temperature) * mean_log_local_prob_pos

        return local_loss

    def avg_global_contrastive_loss(self, features, labels):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        BC, T, D = features.shape

        pooled_features = F.max_pool1d(features.permute(0, 2, 1), kernel_size=T).squeeze(-1) # (BC, D)

        # Compute global representation similarities
        global_features = pooled_features.reshape(self.batch_size, -1, D).permute(1, 0, 2).clone()
        C, B, D = global_features.shape

        anchor_dot_contrast = torch.div(
            torch.bmm(global_features, global_features.transpose(1, 2)),  # (C, B, D) X (C, D, B) > (C, B, B)
            self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        global_logits = anchor_dot_contrast - logits_max.detach()  # subtract most large value

        # tile mask
        global_distmap = (labels.unsqueeze(0) - labels.unsqueeze(1)).abs()  #(B, B)
        global_distmap = global_distmap.reshape(1, B * B).repeat(C, 1).cpu()
        global_distmap = torch.gather(self.acf_values,1, global_distmap).float().to(device)
        global_distmap = global_distmap.reshape(C, B, B)

        neg_mask = torch.scatter(
            torch.ones_like(global_distmap),
            2,
            torch.arange(B).reshape(1, -1, 1).repeat(C, 1, 1).to(device),
            0
        )

        self_mask = (global_distmap == 1.0)

        pos_mask = autocorr_mask_with_CI(global_distmap, self_mask) + (neg_mask * self_mask)

        exp_global_logits = torch.exp(global_logits) * neg_mask  # denominator

        log_global_prob = global_logits - torch.log(exp_global_logits.sum(2, keepdim=True))  # (C, B, B) > (C, B ,1)


        mean_log_global_prob_pos = (global_distmap * pos_mask * log_global_prob).sum(2) \
                                   / pos_mask.sum(2)


        global_loss = - (self.temperature / self.base_temperature) * mean_log_global_prob_pos

        return global_loss

    def forward(self, features, labels=None):

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            print('features shape > 3')

        B, I, D = features.shape
        feature_idxs = torch.rand(B, self.seq_len).argsort(-1)[:, :self.seq_len//3].to(features.get_device())
        selected_features = torch.gather(features, 1, feature_idxs.unsqueeze(-1).repeat(1, 1, D))

        local_loss = self.local_contrastive_loss(selected_features, feature_idxs)
        global_loss = self.avg_global_contrastive_loss(features, labels)

        return local_loss, global_loss
    


def temporal_contrastive_loss(z1, z2, reduction=True):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T，三维矩阵乘法：第一个维度不变，后面的维度相乘
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1),取出下三角部分【torch.tril】
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]#取出上三角部分
    logits = -F.log_softmax(logits, dim=-1)# B x 2T x (2T-1)

    t = torch.arange(T, device=z1.device)#生成0~T-1的整数
    if reduction:
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    else:
        loss = (logits[:, t, T + t - 1].mean(dim=1) + logits[:, T + t, t].mean(dim=1)) / 2
    return loss#, a_loss


def instance_contrastive_loss(z1, z2, reduction=True):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    if reduction:
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    else:
        loss = (logits[:, i, B + i - 1].mean(dim=0) + logits[:, B + i, i].mean(dim=0)) / 2

    return loss#, a_loss


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0, reduction=True):
    B = z1.size(0)
    if reduction:
        loss = torch.tensor(0., device=z1.device)
    else:
        loss = torch.zeros(B, device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2, reduction)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2, reduction)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2, reduction)
        d += 1
    return loss / d


def relative_mask(distance_matrix):
    same_label_mask = (distance_matrix == 0.0)
    relative_matrix = distance_matrix.masked_fill(same_label_mask, np.inf) # remove same label
    min_vals, _ = torch.min(relative_matrix, dim=1, keepdim=True)
    pos_mask = (relative_matrix == min_vals).float()
    neg_mask = torch.ones_like(relative_matrix) - same_label_mask.float()
    return pos_mask, neg_mask


def get_circle_embedding(N):
    index = np.arange(N)
    interval = 2 * np.pi / N
    theta = index * interval
    x = np.cos(theta)
    y = np.sin(theta)
    embeds = np.stack([x, y], axis=1)
    return embeds


def autocorr_mask_with_CI(distance_matrix, self_mask):
    distance_matrix_wo_self = distance_matrix.masked_fill(self_mask, -np.inf) # remove same label
    max_vals, _ = torch.max(distance_matrix_wo_self, dim=2, keepdim=True)
    pos_mask = (distance_matrix_wo_self == max_vals).float()  # max acf is positive pair
    return pos_mask

def autocorr_mask(distance_matrix, self_mask):
    distance_matrix_wo_self = distance_matrix.masked_fill(self_mask, -np.inf) # remove same label
    max_vals, _ = torch.max(distance_matrix_wo_self, dim=1, keepdim=True)
    pos_mask = (distance_matrix_wo_self == max_vals).float()  # max acf is positive pair
    return pos_mask


def local_autocorr_mask(distance_matrix, self_mask):
    distance_matrix_wo_self = distance_matrix.masked_fill(self_mask, -np.inf) # remove same label，True的地方替换为负无穷大，即自己和自己的距离
    max_vals, _ = torch.max(distance_matrix_wo_self, dim=2, keepdim=True)
    pos_mask = (distance_matrix_wo_self == max_vals).float()  # max acf is positive pair，找出对应位置的最大值
    return pos_mask