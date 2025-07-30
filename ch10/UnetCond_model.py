import math
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import random, numpy as np, torch

# ① 共通シード設定関数
def seed_min(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # GPU も一括
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_min(42)

# ② DataLoader の乱数源を固定（shuffle の順序が安定）
gen = torch.Generator().manual_seed(42)

# ハイパーパラメータの設定
img_size = 28
batch_size = 128
num_timesteps = 1000
epochs = 30
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'UnetCond_base_30' # モデル名



# 可視化関数（9章より）
"""
grid表示用の画像を表示する関数（学習後ラベルごとにサンプリング）
    images: 画像のリストまたは配列
    rows: 行数
    cols: 列数
"""
def show_images(images, labels=None, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, i + 1) # 各ラベルごとにサブプロットを作成
            plt.imshow(images[i], cmap='gray') # グレースケールで画像表示
            if labels is not None:
                ax.set_xlabel(labels[i].item()) # ラベルを表示
            ax.get_xaxis().set_ticks([]) # 目盛非表示
            ax.get_yaxis().set_ticks([]) # 目盛非表示
            i += 1
    plt.tight_layout()
    plt.show()

# 正弦波エンコーディング（9章より）

def _pos_encoding(time_idx, output_dim, device='cpu'):
    t, D = time_idx, output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device)
    div_term = torch.exp(i / D * math.log(10000)) # = 10000**(i/D)

    v[0::2] = torch.sin(t / div_term[0::2]) # 偶数次元はsin
    v[1::2] = torch.cos(t / div_term[1::2]) # 奇数次元はcos
    return v

def pos_encoding(timesteps, output_dim, device='cpu'): # バッチ対応版
    batch_size = len(timesteps)
    # device = timesteps.device # テンソルが乗ってるデバイスに変換
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size): # 各バッチでエンコーディング
        v[i] = _pos_encoding(timesteps[i], output_dim, device)
    return v

# Unetをつかった条件付き生成モデルの定義

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), # 3*3畳込み，パディング1，チャネル数変換
            nn.BatchNorm2d(out_ch), # バッチ正規化
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.mlp = nn.Sequential( # 全結合
            nn.Linear(time_embed_dim, in_ch), # 時間埋め込みをチャネル数に変換
            nn.ReLU(),
            nn.Linear(in_ch, in_ch) # チャネル数と同じ次元のベクトルに変換
        )

    def forward(self, x, v): # 入力と時間ベクトルを結合
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y
    
class UNetCond(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100, num_labels=None):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        # 変更箇所①：ラベル処理の埋め込み層（埋め込み＝数値ベクトル変換）
        if num_labels is not None:
            # ラベル数が，time_embed_dimの次元数と同じになるように設定
            self.label_emb = nn.Embedding(num_labels, time_embed_dim)

    def forward(self, x, timesteps, labels=None):
        t = pos_encoding(timesteps, self.time_embed_dim) # t = vのこと

        # 変更箇所②：ラベルがある場合はtに加算して追加処理
        if labels is not None:
            t += self.label_emb(labels)

        x1 = self.down1(x, t)
        x = self.maxpool(x1)
        x2 = self.down2(x, t)
        x = self.maxpool(x2)

        x = self.bot1(x, t)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, t)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, t)
        x = self.out(x)
        return x
    
# Diffusionモデルの定義

class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all() # 1以上T以下の整数であることを確認

        t_idx = t - 1  # alpha_bars[0] is for t=1
        alpha_bar = self.alpha_bars[t_idx]  # (N,)
        alpha_bar = alpha_bar.view(alpha_bar.size(0), 1, 1, 1)  # (N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device) # 元画像x0と同じ形状のノイズテンソル
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise # 変数変換トリック
        return x_t, noise # 時刻tのノイズ画像, ノイズ

    def denoise(self, model, x, t, labels):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all() # 1以上T以下の整数であることを確認

        t_idx = t - 1  # alphas[0] is for t=1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx-1]

        # ブロードキャストのための形状変換
        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()
        with torch.no_grad(): # 評価時には，勾配計算を無効化
            # 変更箇所③：ラベルをモデルに渡す
            eps = model(x, t, labels) # noiseである\epsilonの予測
        model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0  # t=1のときはノイズを0にする（最初の画像はノイズなし）

        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))
        return mu + noise * std

    def reverse_to_img(self, x): # テンソルxからPILイメージに変換（前処理でテンソルにしているため）
        x = x * 255
        x = x.clamp(0, 255) # 0-255の範囲に制限
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage() # テンソルをPIL画像に変換
        return to_pil(x)

    def sample(self, model, x_shape=(20, 1, 28, 28), labels=None): # サンプリング関数（生成）
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        # 変更箇所④：ラベルが指定されていない場合はランダムに生成
        if labels is None:
            labels = torch.randint(0, 10, (len(x),), device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t, labels)

        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images, labels


# ---------------------------------------------- #
preprocess = transforms.ToTensor() # 前処理：テンソル変換
dataset = torchvision.datasets.MNIST(root='./data', download=True, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=gen) # 乱数源を指定

diffuser = Diffuser(num_timesteps, device=device)
# ① モデルをUnetCondに変更
model = UNetCond(num_labels=10)
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)

losses = []
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    # generate samples every epoch ===================
    # images, labels = diffuser.sample(model)
    # show_images(images, labels)
    # ================================================

    for images, labels in tqdm(dataloader):
        optimizer.zero_grad()
        x = images.to(device)
        # ② ラベルをデバイスに転送
        labels = labels.to(device)
        t = torch.randint(1, num_timesteps+1, (len(x),), device=device, generator=gen) # 1からnum_timesteps(T)までのランダムな整数

        x_noisy, noise = diffuser.add_noise(x, t) # ノイズ追加済み画像とノイズを生成
        # ③ ラベルをモデルに渡す
        noise_pred = model(x_noisy, t, labels)
        loss = F.mse_loss(noise, noise_pred) # ノイズ予測と実際のノイズの平均二乗誤差

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f'Epoch {epoch} | Loss: {loss_avg}')

# plot losses
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# generate samples
images, labels = diffuser.sample(model)
show_images(images, labels)

save_path = model_name + '.pth'
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': losses,
}, save_path)
