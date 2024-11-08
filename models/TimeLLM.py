from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # freeze the llm model
        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout) # relieve overfitting

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        # 将 GPT-2 词汇映射到更小的自定义词汇表大小
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        # 重编程层，将嵌入空间映射到自定义维度。
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        # 计算分块数量和用于 FlattenHead 的特征维度
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 输出投影层，按任务（预测任务）将嵌入映射为最终预测输出
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError
        # 归一化层，用于标准化输入特征
        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # 截取 forecast 返回的最后 pred_len 个预测值
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm') # normalization

        # B: batch; T: time series length; N: feature number = 1
        B, T, N = x_enc.size() # (8, 512, 1)
        # shape: x_enc->(8, 1, 512)->(8, 512, 1)
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1) # alloc a contiguous memory for x_enc

        # 做 patch embedding 并将 patch embedding 映射到自定义维度
        # shapes below: (8, 1)
        min_values = torch.min(x_enc, dim=1)[0] # torch.min returns a tuple of (values, index)
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values # torch.median returns the median(tensor type) of the values input
        lags = self.calcute_lags(x_enc) # lags shape: (8, 5), x_enc(8, 512, 1)
        # torch.diff(): Computes the n-th forward difference along the given dimension.
        trends = x_enc.diff(dim=1).sum(dim=1) # 计算差分并求和

        prompt = []
        for b in range(x_enc.shape[0]): # generate a prompt for each input sequence
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_) # total len of prompt = 8

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous() # (8, 512, 1)

        # prompt shpae: list[8] -> tensor.size(8, 127)
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids # tokenize the prompt list

        # 对 prompt 进行分词和嵌入操作，以生成语言模型的输入嵌入。
        # 1. get input embeddings layer; 2. migrate prompt to the device same as x_enc; 3. pass prompt into embeddings layer for forward computing
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim) # a chaining calls in Python
        # transform prompt into embeddings, embedding shape: (8, 126, 768)

        # 从模型的词汇嵌入中提取特征，用于与语言模型嵌入结合。 论文中的 Text Prototype
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0) # 对词嵌入作投影变换
        # print(f'prompt_embeddings:\n{prompt_embeddings}\nshape: {prompt_embeddings.shape}')
        # print(f'source_embeddings:\n{source_embeddings}\nshape: {source_embeddings.shape}')
        # source_embeddings shape: (1000, 768)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # enc_out shape: (8, 64, 32)
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        # llama_enc_out shape: (8, 191, 768)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        # dec_out shape: (8, 191, 768) = (B, L, H)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        # dec_out shape: (8, 191, 128), 保留最后一层隐状态的前 self.d_ff 个特征维度
        dec_out = dec_out[:, :, :self.d_ff]

        # 重塑后的 dec_out 形状为 (批次大小, 特征数量, 时间步长, 特征维度) = (8, 1, 191, 128)
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous() # (批次大小, 特征数量, 特征维度, 时间步长)

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        # 使 dec_out 的形状为 (批次大小, 时间步长, 特征数量)，与时间序列输出的格式一致。
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        # 使用 normalize_layers 的 denorm 模式将输出 dec_out 反归一化，以确保其与输入特征的实际数据范围一致。
        # 这一步可以恢复预测值到原始数值范围，使输出符合实际需求。
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        # torch.fft.rfft(): Computes the one dimensional Fourier transform of real-valued input.
        """
            对输入 x_enc 的最后一维进行快速傅里叶变换（FFT）。
            x_enc.permute(0, 2, 1) 是为了将时间维度置于最后一维以便进行FFT。
            rfft 只计算实数部分的FFT，输出形状为 (batch_size, channels, freq_bins)
        """
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1) # q_fft shape: (8, 1, 257)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        # 将 q_fft 与 k_fft 的共轭相乘，得到自相关频谱。
        res = q_fft * torch.conj(k_fft) # shape: (8, 1, 257)
        # 对 res 进行逆傅里叶变换，得到时域上的自相关性 corr
        corr = torch.fft.irfft(res, dim=-1) # Computes the inverse of rfft().
        # 对相关性结果 corr 进行平均，取每个批次的自相关均值。
        mean_value = torch.mean(corr, dim=1)
        # 取 top_k 个最大值对应的下标作为 lags
        _, lags = torch.topk(mean_value, self.top_k, dim=-1) # return the indices of top_k values
        return lags


"""
    ReprogrammingLayer 类的设计目的是通过注意力机制，将目标嵌入（target_embedding）与
    源嵌入（source_embedding）和数值嵌入（value_embedding）重新编程和整合。
    这一层使用多头注意力机制来生成一个新的嵌入表示，并将其输出到下游的 d_llm 维度空间中。
"""
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads
        # (B: batch_size, L: seq_len, H: num_heads, d_keys: num_features_per_head)
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        # 计算注意力输出, out shape: (B, L, H, d_keys) = (8, 64, 8, 128)
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        # out shape: (B, L, H * d_keys) = (8, 64, 1024)
        out = out.reshape(B, L, -1)

        # 将输出通过 out_projection 映射到 d_llm 维度空间。
        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        # 缩放因子使注意力分数更稳定，以避免数值过大导致的梯度消失或爆炸。这是多头注意力中常用的缩放方法。
        scale = 1. / sqrt(E)
        # 使用 torch.einsum 计算注意力得分，其中 blhe 和 she 分别表示目标嵌入和源嵌入的形状。
        # 这一步得到的 scores 的维度是 (B, H, L, S)，即每个头上每对 target_embedding 和 source_embedding 之间的相似度。
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        # 对得分进行缩放后，使用 softmax 归一化，生成注意力矩阵 A。使用 dropout 随机失活一些注意力权重，减少模型过拟合。
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # 用注意力矩阵 A 加权 value_embedding，得到重新组合后的嵌入。
        # 结果的形状为 (B, L, H, E)，即保留了输入 target_embedding 的维度结构。
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
