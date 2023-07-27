# -*- encoding: utf-8 -*-
'''
Filename         :model.py
Description      :
Time             :2023/07/25 08:49:18
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
from typing import Text, Any
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    def __init__(self, 
                 input_size:int,
                 encoder_hidden_size:int,
                 encoder_num_layers:int,
                 encodr_dropout_rate:float=0.5,
                 encoder_bidirectional:bool=False,
                 encoder_rnn_type:Text="gru") -> None:
        super().__init__()
        if encoder_rnn_type.lower() == "gru":
            self.encoder_rnn = nn.GRU(
                input_size=input_size,
                hidden_size=encoder_hidden_size,
                num_layers=encoder_num_layers,
                batch_first=True,
                dropout=encodr_dropout_rate if encoder_num_layers > 1 else 0,
                bidirectional=encoder_bidirectional
            )
        else:
            raise ValueError(
                "Invalid q_cell type, should be one of the ('gru',)"
            )
        
    def forward(self, x, h=None):
        return self.encoder_rnn(x, h)
    
    
class Decoder(nn.Module):
    def __init__(self, 
                 input_size:int,
                 decoder_hidden_size:int,
                 decoder_num_layers:int,
                 decoder_dropout_arte:float=0.5,
                 decoder_bidirectional:bool=False,
                 decoder_rnn_type: Text = "gru",
                 ) -> None:
        super().__init__()
        if decoder_rnn_type == "gru":
            self.decoder_rnn = nn.GRU(
                input_size=input_size,
                hidden_size=decoder_hidden_size,
                num_layers=decoder_num_layers,
                batch_first=True,
                dropout=decoder_dropout_arte if decoder_num_layers > 1 else 0,
                bidirectional=decoder_bidirectional
            )
        else:
            raise ValueError(
                "Invalid d_cell type, should be one of the ('gru',)"
            )
    
    def forward(self, x, h):
        return self.decoder_rnn(x, h)

class VAE(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 encoder_hidden_size: int,
                 encoder_num_layers: int,
                 encoder_bidirectional: bool,
                 encoder_z_liner_dim: int,
                 decoder_hidden_size: int,
                 decoder_num_layers: int,
                 decoder_bidirectional: bool,
                 decoder_z_liner_dim: int,
                 encodr_dropout_rate: float = 0.5,
                 decoder_dropout_arte: float = 0.5,
                 pad_token_ids: int = 0,
                 encoder_rnn_type: Text = "gru",
                 decoder_rnn_type: Text = "gru",
                 freeze_embeddings: bool = False
                 ):
        super(VAE, self).__init__()
        self.pad_token_ids = pad_token_ids

        vectors = torch.eye(vocab_size)
        embedding_dim = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_ids)
        self.embedding.weight.data.copy_(vectors)

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.encoder = Encoder(input_size=embedding_dim,
                              encoder_hidden_size=encoder_hidden_size,
                              encoder_num_layers=encoder_num_layers,
                              encodr_dropout_rate=encodr_dropout_rate if encoder_num_layers > 1 else 0,
                              encoder_bidirectional=encoder_bidirectional,
                              encoder_rnn_type=encoder_rnn_type,
                              )
       
        ##参数重建 mu sigma esp
        q_encoder_last = encoder_hidden_size * (2 if encoder_bidirectional else 1)
        self.q_mu = nn.Linear(q_encoder_last, encoder_z_liner_dim)
        self.q_logvar = nn.Linear(q_encoder_last, encoder_z_liner_dim)

        ## 解码
        self.decoder = Decoder(input_size=embedding_dim + encoder_z_liner_dim,
                               decoder_hidden_size=decoder_hidden_size,
                               decoder_num_layers=decoder_num_layers,
                               decoder_dropout_arte= decoder_dropout_arte,
                               decoder_bidirectional=decoder_bidirectional,
                               decoder_rnn_type=decoder_rnn_type
                               )
        ## 重建后参数映射
        self.decoder_lat = nn.Linear(encoder_z_liner_dim, decoder_z_liner_dim)
        decoder_z_liner_dim=2*decoder_z_liner_dim if self.decoder.decoder_rnn.bidirectional else decoder_z_liner_dim
        self.decoder_fc = nn.Linear(decoder_z_liner_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        z, kl_loss = self.forward_encoder(input_ids=input_ids)
        recon_loss = self.forward_decoder(input_ids=input_ids, z=z)
        return kl_loss, recon_loss

    def forward_encoder(self,
                        input_ids: torch.Tensor):
        x = [self.embedding(x) for x in input_ids]
        x = nn.utils.rnn.pack_sequence(x)
        _, h = self.encoder(x, None)  # [bz, seq_len, hz] [D∗num_layers, N, Hout​]
        h = h[-(1 + int(self.encoder.encoder_rnn.bidirectional)):]  # (D∗num_layers, N, Hout​)
        h = torch.cat(h.split(1), dim=-1).squeeze(0)  # [N, D*num_layer*hz]
        mu, logvar = self.q_mu(h), self.q_logvar(h)  # 学习均值和方差
        eps = torch.randn_like(mu)  # 返回一个和输入大小相同的张量，其由均值为0、方差为1的标准正态分布填充
        z = mu + (logvar / 2).exp() * eps  # z分布~p(z)
        # 公式推导： log(\frac{\sigma_2}{\sigma_1}) + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
        # 假设N2是一个正态分布，也就是说\mu_2=0, \sigma_2^2=1, 也就是说KL(\mu_1, \sigma_1) = -log(\sigma_1) + \frac{\sigma_1^2+\mu_1^2}{2} - \frac{1}{2}
        # -->  kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # 参考来自： https://blog.csdn.net/qq_31895943/article/details/90754390
        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()  # dl(p(z|x)||p(z))
        return z, kl_loss

    def forward_decoder(self,
                        input_ids: torch.Tensor,
                        z: torch.Tensor):
        lengths = [len(i_x) for i_x in input_ids]
        input_ids = nn.utils.rnn.pad_sequence(input_ids,
                                              batch_first=True,
                                              padding_value=self.pad_token_ids)
        x_emb = self.embedding(input_ids)
        # 当参数只有两个时：（列的重复倍数，行的重复倍数）。
        # 1表示不重复  当参数有三个时：（通道数的重复倍数，列的重复倍数，行的重复倍数）。
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths,
                                                    batch_first=True)

        h_0 = self.decoder_lat(z)
        num_repeat = 2*self.decoder.decoder_rnn.num_layers if self.decoder.decoder_rnn.bidirectional else self.decoder.decoder_rnn.num_layers
        h_0 = h_0.unsqueeze(0).repeat(num_repeat, 1, 1)

        output, _ = self.decoder(x_input, h_0)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=self.pad_token_ids)
        y = self.decoder_fc(output)

        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            input_ids[:, 1:].contiguous().view(-1),
            ignore_index=self.pad_token_ids
        )  # MSE
        return recon_loss
    
    
    def sample(self, 
               batch:int, 
               max_length:int, 
               bos_id:int,
               pad_id:int,
               eos_id:int,
               z:Any=None, 
               temp:float=1.0):
        device = self.embedding.weight.device
        ## z 是随机值，或者是内容值
        if z == None:
            z = torch.randn(batch, self.q_mu.out_features,device=device) # [bz, hz]
        
        z_0 = z.unsqueeze(1) #[bz, 1, hz]
        h = self.decoder_lat(z) # [bz, hz]
        h = h.unsqueeze(0).repeat(self.decoder.decoder_rnn.num_layers*(2 if self.decoder.decoder_rnn.bidirectional else 1), 1, 1)
        w = torch.tensor(bos_id,device=device).repeat(batch)
        x = torch.tensor(pad_id, device=device).repeat(batch, max_length)
        x[:, 0] = bos_id
        end_pads = torch.tensor([max_length], device=device).repeat(batch)
        eos_mask = torch.zeros(batch, device=device, dtype=torch.uint8).bool()
        
        # Generating cycle
        for i in range(1, max_length):
            x_emb = self.embedding(w).unsqueeze(1)
            x_input = torch.cat([x_emb, z_0], dim=-1)
            o, h = self.decoder(x_input, h)
            y = self.decoder_fc(o.squeeze())
            y = F.softmax(y/temp, dim=-1)
            ## 采样
            w = torch.multinomial(y, 1)[:, 0]
            x[~eos_mask, i] = w[~eos_mask]
            i_eos_mask = ~eos_mask & (w==eos_id)# 新增加True
            end_pads[i_eos_mask] = i+1
            eos_mask = eos_mask | i_eos_mask 
        
        new_x = []
        for i in range(batch):
            new_x.append(x[i, :end_pads[i]])    
        return new_x
if __name__ == '__main__':
    import torch
    inputx = torch.randint(0, 30, size=(512, 49))
    model = VAE(vocab_size=30,
                encoder_hidden_size=128,
                encoder_num_layers=1,
                encoder_bidirectional=False,
                encoder_z_liner_dim=128,
                decoder_hidden_size=512,
                decoder_num_layers=3,
                decoder_bidirectional=True,
                decoder_z_liner_dim=512,
                pad_token_ids=28)
    res = model(inputx)
    model.sample(batch=32, max_length=100, bos_id=26, pad_id=28, eos_id=27)
    print(res)
