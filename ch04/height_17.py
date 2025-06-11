import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# CSV読み込み
df = pd.read_csv("./data/学校保健統計調査_身長の体格別分布.csv", encoding="utf-8")
df["身長"] = df["身長（90～200cm）"].str.extract(r"(\d+)").astype(float)
df["17歳_分布"] = pd.to_numeric(df["高等学校（17歳）"], errors="coerce")

# 男女に分割
df_male = df[df["性別"] == "男"].dropna(subset=["17歳_分布"]).copy()
df_female = df[df["性別"] == "女"].dropna(subset=["17歳_分布"]).copy()
df_male["prob"] = df_male["17歳_分布"] / 1000
df_female["prob"] = df_female["17歳_分布"] / 1000

# 平均・標準偏差（加重）
mu_m = np.average(df_male["身長"], weights=df_male["prob"])
sigma_m = np.sqrt(np.average((df_male["身長"] - mu_m)**2, weights=df_male["prob"]))
mu_f = np.average(df_female["身長"], weights=df_female["prob"])
sigma_f = np.sqrt(np.average((df_female["身長"] - mu_f)**2, weights=df_female["prob"]))

# 正規分布
x = np.linspace(140, 190, 500)
pdf_m = norm.pdf(x, mu_m, sigma_m)
pdf_f = norm.pdf(x, mu_f, sigma_f)

# ✅ ここで定義！
bar_width = 0.8
offset = 0.4

# プロット
plt.figure(figsize=(10, 6))
plt.bar(df_male["身長"] - offset, df_male["prob"], width=bar_width, alpha=0.6,
        label="boy (data)", color="blue", edgecolor="black")
plt.bar(df_female["身長"] + offset, df_female["prob"], width=bar_width, alpha=0.6,
        label="girl (data)", color="red", edgecolor="black")
plt.plot(x, pdf_m, '-', color="blue", label=f"boy Gaussian\nμ={mu_m:.2f}, σ={sigma_m:.2f}")
plt.plot(x, pdf_f, '--', color="red", label=f"girl Gaussian\nμ={mu_f:.2f}, σ={sigma_f:.2f}")
plt.title("Height distribution of 17-year-olds by gender and normal distribution fit")
plt.xlabel("heights (cm)")
plt.ylabel("probability")
plt.grid(True)
plt.legend()
plt.tight_layout()

# 保存 & 表示
plt.savefig("height_distribution_17yo.png", dpi=300)
plt.show()