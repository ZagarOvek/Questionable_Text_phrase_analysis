import spacy
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ПАРАМЕТРИ ЩО ШЕПОЧУТЬ МЕНІ В ТЕМРЯВІ
MAX_WORDS_PER_AUTHOR = 10000  # ВОНИ СКАЗАЛИ ЗУПИНИТИСЯ НА 10000
TEXT_DIR = r"D:\Soft\PyCharm Projects\Квантитативна Лангустика\Параметризація\Лабораторна №4"  # МІСЦЕ ДЕ ЖИВУТЬ СЛОВА
OUTPUT_DIR = "results"  # ПАПКА-СХОВИЩЕ ДЛЯ РЕЗУЛЬТАТІВ
os.makedirs(OUTPUT_DIR, exist_ok=True)  # СТВОРЮЄМО СХОВИЩЕ ЯКЩО ЙОГО НЕМАЄ

# СЛОВА-ПРИМАРИ ЯКІ ТРЕБА ВИГНАТИ
STOP_WORDS = {'бути', 'мати', 'робити', 'ставати', 'говорити', 'той', 'цей', 'такий', 'який', 'весь'}

# НАЛАШТУВАННЯ ВІЗУАЛІЗАЦІЇ ДЛЯ ОЧЕЙ ЯКІ БАЧАТЬ ЗАБАГАТО
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

nlp = spacy.load("uk_core_news_lg")  # ЗВІР ПРОКИНУВСЯ


def extract_phrases(doc):
    """ВИТЯГУЄМО СЛОВОСПОЛУЧЕННЯ З ТЕМРЯВИ (вони шепочуться між собою)"""
    phrases = {'verbal': [], 'nominal': [], 'adverbial': []}

    for token in doc:
        # ДІЄСЛОВА ШУКАЮТЬ ДРУЗІВ (їм потрібні іменники та прислівники)
        if token.pos_ == "VERB":
            for child in token.children:
                # ДІЄСЛОВО + ІМЕННИК (вони тримаються за руки в синтаксичному дереві)
                if child.pos_ == "NOUN" and child.dep_ in ["obj", "iobj", "obl", "nsubj", "nmod"]:
                    phrases['verbal'].append(f"{token.lemma_} {child.lemma_}")
                # ДІЄСЛОВО + ПРИСЛІВНИК (тихий шепіт у темряві)
                elif child.pos_ == "ADV" and child.dep_ == "advmod":
                    phrases['verbal'].append(f"{token.lemma_} {child.lemma_}")

            # ЯКЩО ДІЄСЛОВО СЛУХАЄ ІМЕННИК (інверсія влади)
            if token.head.pos_ == "NOUN":
                phrases['verbal'].append(f"{token.lemma_} {token.head.lemma_}")

        # ІМЕННИКИ ТАНЦЮЮТЬ З ПРИКМЕТНИКАМИ (номінальний бал)
        if token.pos_ == "NOUN":
            for child in token.children:
                # ПРИКМЕТНИК + ІМЕННИК (класична пара)
                if child.pos_ == "ADJ":
                    phrases['nominal'].append(f"{child.lemma_} {token.lemma_}")
                # ІМЕННИК + ІМЕННИК (заборонений союз)
                elif child.pos_ == "NOUN":
                    phrases['nominal'].append(f"{token.lemma_} {child.lemma_}")

            # ЗВОРОТНИЙ ЗВ'ЯЗОК (якщо іменник підпорядковується прикметнику)
            if token.head.pos_ == "ADJ":
                phrases['nominal'].append(f"{token.head.lemma_} {token.lemma_}")

        # ПРИСЛІВНИКИ ХОВАЮТЬСЯ В ТІНІ (модифікують усіх підряд)
        if token.pos_ == "ADV":
            # ПРИСЛІВНИК ШЕПОЧЕ ДІЄСЛОВУ
            if token.head.pos_ == "VERB":
                phrases['adverbial'].append(f"{token.lemma_} {token.head.lemma_}")
            # ПРИСЛІВНИК ТОРКАЄТЬСЯ ПРИКМЕТНИКА
            elif token.head.pos_ == "ADJ":
                phrases['adverbial'].append(f"{token.lemma_} {token.head.lemma_}")
            # ПРИСЛІВНИК РОЗМОВЛЯЄ З ІНШИМ ПРИСЛІВНИКОМ (луна в порожнечі)
            elif token.head.pos_ == "ADV":
                phrases['adverbial'].append(f"{token.lemma_} {token.head.lemma_}")

    return phrases


def filter_phrases(phrases_dict):
    """ФІЛЬТРУЄМО ШЕПІТ ВІД БРЕХЛИВИХ СЛІВ (стоп-слова та повтори)"""
    filtered = {'verbal': [], 'nominal': [], 'adverbial': []}

    for phrase_type, phrase_list in phrases_dict.items():
        for phrase in phrase_list:
            words = phrase.split()
            # ПЕРЕВІРЯЄМО НА ПРИМАРІВ І КАРЛИКІВ
            if len(words) == 2:
                # ВИГНАННЯ ПОВТОРІВ (заїкаючіся слова)
                if words[0] == words[1]:
                    continue
                # ВИГНАННЯ СТОП-СЛІВ І ЗАКОРОТКИХ (вони спостерігають за нами)
                if (words[0] not in STOP_WORDS and words[1] not in STOP_WORDS and
                        len(words[0]) > 2 and len(words[1]) > 2):
                    filtered[phrase_type].append(phrase)

    return filtered


# === ЧИТАННЯ ТА ОБРОБКА ТЕКСТІВ (вони приходять із темряви) ===
print("📚 ЗАВАНТАЖУЄМО ТЕКСТИ... (слова вишиковуються в ряди)")
all_data = []
author_stats = {}

for file in os.listdir(TEXT_DIR):
    if file.endswith(".txt"):  # ТЕКСТОВІ ФАЙЛИ - ЦЕ ПОРТАЛИ
        author = os.path.splitext(file)[0]
        print(f"  ОБРОБЛЯЄМО: {author} (ще одна жертва)")

        with open(os.path.join(TEXT_DIR, file), encoding="utf-8") as f:
            words = f.read().split()[:MAX_WORDS_PER_AUTHOR]  # ПІДРІЗАЄМО КРИЛА СЛІВАМ
            text = " ".join(words)  # З'ЄДНУЄМО ЇХ У ЛАНЦЮЖОК

        doc = nlp(text)  # ПРОПУСКАЄМО КРІЗЬ МЛИН
        phrases = extract_phrases(doc)  # ВИТЯГУЄМО СПОЛУЧЕННЯ
        phrases = filter_phrases(phrases)  # ФІЛЬТРУЄМО ШУМ

        # СТАТИСТИКА ДЛЯ БОГІВ ДАНИХ
        author_stats[author] = {
            'words': len(words),  # СКІЛЬКИ СЛІВ ПОБАЧИВ
            'verbal': len(phrases['verbal']),  # ДІЄСЛІВНІ ШЕПОТИ
            'nominal': len(phrases['nominal']),  # ІМЕННІ ТІНІ
            'adverbial': len(phrases['adverbial']),  # ПРИСЛІВНИКОВІ ЛУНИ
            'total': sum(len(p) for p in phrases.values())  # ВСЕ РАЗОМ
        }

        for typ, items in phrases.items():
            for phrase in items:
                all_data.append({"author": author, "type": typ, "phrase": phrase})

df = pd.DataFrame(all_data)  # ПЕРЕТВОРЮЄМО ХАОС НА ТАБЛИЦЮ
print(f"\n✅ ЗНАЙДЕНО {len(df)} СЛОВОСПОЛУЧЕНЬ ВІД {len(author_stats)} АВТОРІВ (вони серед нас)")

# ВИВОДИМО СТАТИСТИКУ (цифри шепочуть правду)
print("\n📊 СТАТИСТИКА ПО АВТОРАХ:")
stats_df = pd.DataFrame(author_stats).T
print(stats_df)
stats_df.to_excel(f"{OUTPUT_DIR}/statistics.xlsx")  # ЗБЕРІГАЄМО ДЛЯ МАЙБУТНІХ ПОКОЛІНЬ

# === ЧАСТОТНИЙ АНАЛІЗ (рахуємо шепіт) ===
print("\n🔢 ЧАСТОТНИЙ АНАЛІЗ... (слова мерехтять з різною частотою)")
freq_df = (
    df.groupby(["author", "type", "phrase"])
    .size()
    .reset_index(name="count")
    .sort_values(["author", "type", "count"], ascending=[True, True, False])
)

# ЗБЕРІГАЄМО В EXCEL (кладемо в різні кімнати)
with pd.ExcelWriter(f"{OUTPUT_DIR}/phrases_by_type.xlsx", engine="openpyxl") as writer:
    for t in ["verbal", "nominal", "adverbial"]:
        freq_df[freq_df["type"] == t].to_excel(writer, sheet_name=t, index=False)

# === АНАЛІЗ УНІКАЛЬНОСТІ (шукаємо відбитки пальців) ===
print("\n🔍 АНАЛІЗ УНІКАЛЬНОСТІ... (у кожного автора свій почерк)")
authors = df['author'].unique()
uniqueness_data = []

for phrase_type in ["verbal", "nominal", "adverbial"]:
    type_df = freq_df[freq_df["type"] == phrase_type]

    for author in authors:
        author_phrases = set(type_df[type_df["author"] == author]["phrase"])  # СЛОВА АВТОРА
        other_phrases = set(type_df[type_df["author"] != author]["phrase"])   # СЛОВА ІНШИХ

        unique = len(author_phrases - other_phrases)    # УНІКАЛЬНІ ШЕПОТИ
        shared = len(author_phrases & other_phrases)    # СПІЛЬНІ ГОЛОСИ
        total = len(author_phrases)                     # ВСЕ РАЗОМ

        uniqueness_data.append({
            "author": author,
            "type": phrase_type,
            "unique": unique,
            "shared": shared,
            "total": total,
            "unique_pct": (unique / total * 100) if total > 0 else 0  # ВІДСОТОК УНІКАЛЬНОСТІ
        })

uniqueness_df = pd.DataFrame(uniqueness_data)

# ГРАФІК УНІКАЛЬНОСТІ (візуалізуємо роздвоєння особистості)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
types = ["verbal", "nominal", "adverbial"]
type_labels = ["ДІЄСЛІВНІ", "ІМЕННІ", "ПРИСЛІВНИКОВІ"]

for idx, (t, label) in enumerate(zip(types, type_labels)):
    data = uniqueness_df[uniqueness_df["type"] == t]
    x = np.arange(len(data))
    width = 0.35

    axes[idx].bar(x - width / 2, data["unique"], width, label='УНІКАЛЬНІ', alpha=0.8)
    axes[idx].bar(x + width / 2, data["shared"], width, label='СПІЛЬНІ', alpha=0.8)

    axes[idx].set_xlabel('АВТОР')
    axes[idx].set_ylabel('КІЛЬКІСТЬ')
    axes[idx].set_title(f'{label} СЛОВОСПОЛУЧЕННЯ (шепіт і луна)')
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels([a.split('_')[0] for a in data["author"]], rotation=45, ha='right')
    axes[idx].legend()
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/uniqueness_analysis.png", bbox_inches='tight')
plt.close()

# === ТОП-10 СЛОВОСПОЛУЧЕНЬ (найгучніші шепоти) ===
print("\n📊 СТВОРЮЄМО ГРАФІКИ ТОП-10... (голоси стають гучнішими)")
for t, label in zip(types, type_labels):
    fig, axes = plt.subplots(1, len(authors), figsize=(12, 6), sharey=True)
    if len(authors) == 1:
        axes = [axes]

    for idx, author in enumerate(authors):
        top10 = (
            freq_df[(freq_df["type"] == t) & (freq_df["author"] == author)]
            .nlargest(10, "count")
            .sort_values("count", ascending=True)
        )

        colors = sns.color_palette("viridis", len(top10))
        axes[idx].barh(range(len(top10)), top10["count"], color=colors)
        axes[idx].set_yticks(range(len(top10)))
        axes[idx].set_yticklabels(top10["phrase"], fontsize=9)
        axes[idx].set_xlabel('ЧАСТОТА')
        axes[idx].set_title(author.split('_')[0], fontsize=10, fontweight='bold')
        axes[idx].grid(axis='x', alpha=0.3)

        # ДОДАЄМО ЦИФРИ (вони кричать з вершин)
        for i, v in enumerate(top10["count"]):
            axes[idx].text(v, i, f' {v}', va='center', fontsize=8)

    axes[0].set_ylabel('СЛОВОСПОЛУЧЕННЯ')
    fig.suptitle(f'ТОП-10 {label} ШЕПОТІВ', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/top10_{t}.png", bbox_inches='tight')
    plt.close()

# === ВЕКТОРИЗАЦІЯ (перетворюємо слова на числа) ===
print("\n🧮 ВЕКТОРНИЙ АНАЛІЗ... (слова стають точками у просторі)")
sentences = [p.split() for p in df["phrase"]]
model = Word2Vec(sentences, vector_size=100, window=3, min_count=1, workers=4, epochs=20)  # НАВЧАЄМО МОЗОК


def phrase_vector(phrase):
    """ПЕРЕТВОРЮЄМО ФРАЗУ НА ВЕКТОР (слово стає числом, число стає долею)"""
    words = phrase.split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else None


df["vector"] = df["phrase"].apply(phrase_vector)
df = df.dropna(subset=["vector"])  # ВИДАЛЯЄМО ПУСТОТУ

# === PCA ВІЗУАЛІЗАЦІЯ (проекція багатовимірного кошмару) ===
print("\n📉 PCA ВІЗУАЛІЗАЦІЯ... (стискаємо виміри)")
agg_df = (
    df.groupby(["author", "type"])["vector"]
    .apply(lambda v: np.mean(np.vstack(v), axis=0))
    .reset_index()
)

pca = PCA(n_components=2)  # ЗВОДИМО ДО 2-Х ВИМІРІВ
coords = pca.fit_transform(list(agg_df["vector"]))
agg_df["x"], agg_df["y"] = coords[:, 0], coords[:, 1]

# ГРАФІК PCA (карта зоряного неба зі слів)
plt.figure(figsize=(10, 7))
markers = {'verbal': 'o', 'nominal': 's', 'adverbial': '^'}
colors = sns.color_palette("Set2", len(authors))

for idx, author in enumerate(authors):
    author_data = agg_df[agg_df["author"] == author]
    for ptype in ['verbal', 'nominal', 'adverbial']:
        type_data = author_data[author_data["type"] == ptype]
        if not type_data.empty:
            plt.scatter(
                type_data["x"], type_data["y"],
                marker=markers[ptype],
                s=300,
                color=colors[idx],
                label=f"{author.split('_')[0]} - {ptype}",
                edgecolors='black',
                linewidths=1.5,
                alpha=0.7
            )

            # ПІДПИСИ (мітки на карті реальності)
            for _, row in type_data.iterrows():
                plt.annotate(
                    ptype[:3],
                    (row["x"], row["y"]),
                    fontsize=8,
                    ha='center',
                    va='center',
                    fontweight='bold'
                )

plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.1%} ДИСПЕРСІЇ)', fontsize=12)
plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.1%} ДИСПЕРСІЇ)', fontsize=12)
plt.title('ВЕКТОРНІ ВІДСТАНІ МІЖ ТИПАМИ СЛІВ', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pca_analysis.png", bbox_inches='tight')
plt.close()

# === ТЕПЛОВА КАРТА СХОЖОСТІ (матриця зв'язків) ===
print("\n🔥 ТЕПЛОВА КАРТА СХОЖОСТІ... (вимірюємо родинність душ)")
vectors_matrix = np.vstack(agg_df["vector"].values)
similarity = cosine_similarity(vectors_matrix)  # КОСИНУСНА РОДИННІСТЬ

labels = [f"{row['author'].split('_')[0]}\n{row['type'][:3]}"
          for _, row in agg_df.iterrows()]

plt.figure(figsize=(10, 8))
sns.heatmap(
    similarity,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    xticklabels=labels,
    yticklabels=labels,
    cbar_kws={'label': 'КОСИНУСНА СХОЖІСТЬ'},
    square=True,
    linewidths=0.5
)
plt.title('СХОЖІСТЬ ВЕКТОРНИХ ПРЕДСТАВЛЕНЬ', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/similarity_heatmap.png", bbox_inches='tight')
plt.close()

# === РОЗПОДІЛ ЧАСТОТ (частотний спектр) ===
print("\n📈 РОЗПОДІЛ ЧАСТОТ... (ритми та патерни)")
fig, axes = plt.subplots(len(authors), 3, figsize=(15, 5 * len(authors)))
if len(authors) == 1:
    axes = axes.reshape(1, -1)

for row_idx, author in enumerate(authors):
    for col_idx, (t, label) in enumerate(zip(types, type_labels)):
        ax = axes[row_idx, col_idx]

        type_freq = freq_df[(freq_df["author"] == author) & (freq_df["type"] == t)]
        freq_counts = Counter(type_freq["count"])

        x = sorted(freq_counts.keys())
        y = [freq_counts[k] for k in x]

        ax.bar(x, y, color=sns.color_palette("muted")[col_idx], alpha=0.7, edgecolor='black')
        ax.set_xlabel('ЧАСТОТА СЛОВОСПОЛУЧЕННЯ')
        ax.set_ylabel('КІЛЬКІСТЬ УНІКАЛЬНИХ ФРАЗ')
        ax.set_title(f'{author.split("_")[0]} - {label}', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/frequency_distribution.png", bbox_inches='tight')
plt.close()

# === ДОДАТКОВИЙ АНАЛІЗ (глибини божевілля) ===
print("\n📊 ДОДАТКОВИЙ СТАТИСТИЧНИЙ АНАЛІЗ... (копаємо глибше)")

# СЕРЕДНІ ЧАСТОТИ (баланс сил)
avg_freq_data = []
for author in authors:
    for ptype in types:
        type_freq = freq_df[(freq_df["author"] == author) & (freq_df["type"] == ptype)]
        if len(type_freq) > 0:
            avg_freq_data.append({
                "author": author,
                "type": ptype,
                "mean_freq": type_freq["count"].mean(),
                "median_freq": type_freq["count"].median(),
                "max_freq": type_freq["count"].max(),
                "unique_phrases": len(type_freq)
            })

avg_freq_df = pd.DataFrame(avg_freq_data)

# ГРАФІК СЕРЕДНІХ ЧАСТОТ
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
type_labels = ["ДІЄСЛІВНІ", "ІМЕННІ", "ПРИСЛІВНИКОВІ"]

for idx, (t, label) in enumerate(zip(types, type_labels)):
    data = avg_freq_df[avg_freq_df["type"] == t]
    x = np.arange(len(data))
    width = 0.25

    axes[idx].bar(x - width, data["mean_freq"], width, label='СЕРЕДНЯ', alpha=0.8)
    axes[idx].bar(x, data["median_freq"], width, label='МЕДІАНА', alpha=0.8)
    axes[idx].bar(x + width, data["max_freq"], width, label='МАКСИМУМ', alpha=0.8)

    axes[idx].set_xlabel('АВТОР')
    axes[idx].set_ylabel('ЧАСТОТА')
    axes[idx].set_title(f'{label} (три обличчя частоти)')
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels([a.split('_')[0] for a in data["author"]], rotation=45, ha='right')
    axes[idx].legend()
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/average_frequencies.png", bbox_inches='tight')
plt.close()

# ІНДЕКС ЖАККАРА (вимірюємо перетин реальностей)
print("\n🔍 РОЗРАХУНОК ІНДЕКСУ ЖАККАРА... (спільні сни)")
jaccard_data = []

for ptype in types:
    if len(authors) >= 2:
        author1_phrases = set(freq_df[(freq_df["author"] == authors[0]) & (freq_df["type"] == ptype)]["phrase"])
        author2_phrases = set(freq_df[(freq_df["author"] == authors[1]) & (freq_df["type"] == ptype)]["phrase"])

        intersection = len(author1_phrases & author2_phrases)  # ПЕРЕТИН СВІТІВ
        union = len(author1_phrases | author2_phrases)        # ОБ'ЄДНАНА РЕАЛЬНІСТЬ
        jaccard = intersection / union if union > 0 else 0    # КОЕФІЦІЄНТ РОДИННОСТІ

        jaccard_data.append({
            "type": ptype,
            "author1_unique": len(author1_phrases - author2_phrases),  # УНІКАЛЬНІ СНИ
            "author2_unique": len(author2_phrases - author1_phrases),
            "shared": intersection,                                    # СПІЛЬНІ БАЧЕННЯ
            "jaccard_index": jaccard,
            "overlap_pct": (intersection / min(len(author1_phrases), len(author2_phrases)) * 100) if min(
                len(author1_phrases), len(author2_phrases)) > 0 else 0  # ВІДСОТОК ПЕРЕКРИТТЯ
        })

jaccard_df = pd.DataFrame(jaccard_data)

# ГРАФІК ІНДЕКСУ ЖАККАРА
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# СТОВПЧИКИ ІНДЕКСУ ЖАККАРА
x = np.arange(len(jaccard_df))
colors = sns.color_palette("viridis", len(jaccard_df))
bars = ax1.bar(x, jaccard_df["jaccard_index"], color=colors, alpha=0.7, edgecolor='black')
ax1.set_xlabel('ТИП СЛОВОСПОЛУЧЕНЬ')
ax1.set_ylabel('ІНДЕКС ЖАККАРА')
ax1.set_title('СХОЖІСТЬ СЛОВНИКІВ МІЖ АВТОРАМИ\n(ІНДЕКС СПІЛЬНИХ СНІВ)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(['ДІЄСЛІВНІ', 'ІМЕННІ', 'ПРИСЛІВНИКОВІ'])
ax1.set_ylim(0, 1)
ax1.grid(axis='y', alpha=0.3)

# ДОДАЄМО ЗНАЧЕННЯ (цифри на стовпчиках)
for i, (bar, val) in enumerate(zip(bars, jaccard_df["jaccard_index"])):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# СТОВПЧИКИ УНІКАЛЬНИХ VS СПІЛЬНИХ
type_names = ['ДІЄСЛІВНІ', 'ІМЕННІ', 'ПРИСЛІВНИКОВІ']
x2 = np.arange(len(jaccard_df))
width = 0.25

bars1 = ax2.bar(x2 - width, jaccard_df["author1_unique"], width,
                label=authors[0].split('_')[0], alpha=0.8)
bars2 = ax2.bar(x2, jaccard_df["shared"], width,
                label='СПІЛЬНІ', alpha=0.8)
bars3 = ax2.bar(x2 + width, jaccard_df["author2_unique"], width,
                label=authors[1].split('_')[0], alpha=0.8)

ax2.set_xlabel('ТИП СЛОВОСПОЛУЧЕНЬ')
ax2.set_ylabel('КІЛЬКІСТЬ ФРАЗ')
ax2.set_title('РОЗПОДІЛ УНІКАЛЬНИХ ТА СПІЛЬНИХ ФРАЗ', fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(type_names)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/jaccard_analysis.png", bbox_inches='tight')
plt.close()

# ЗБЕРІГАЄМО СТАТИСТИКУ (для майбутніх дослідників божевілля)
avg_freq_df.to_excel(f"{OUTPUT_DIR}/frequency_statistics.xlsx", index=False)
jaccard_df.to_excel(f"{OUTPUT_DIR}/jaccard_statistics.xlsx", index=False)

print("\n✅ ГОТОВО! РЕЗУЛЬТАТИ ЗБЕРЕЖЕНО В ПАПЦІ 'results/':")
print("   - statistics.xlsx (загальна статистика)")
print("   - phrases_by_type.xlsx (словники)")
print("   - uniqueness_analysis.png (унікальність)")
print("   - top10_*.png (топ словосполучення)")
print("   - pca_analysis.png (PCA)")
print("   - similarity_heatmap.png (схожість векторів)")
print("   - frequency_distribution.png (розподіл частот)")
print("   - average_frequencies.png (середні частоти)")
print("   - jaccard_analysis.png (індекс Жаккара)")
print("   - frequency_statistics.xlsx (статистика частот)")
print("   - jaccard_statistics.xlsx (статистика схожості)")
print("\n📊 РЕЗЮМЕ:")
print(f"   ВСЬОГО ЗНАЙДЕНО: {len(df)} словосполучень (голосів з темряви)")
print(f"   УНІКАЛЬНИХ ФРАЗ: {len(freq_df)} (індивідуальних почерків)")
print("\n   ІНДЕКСИ ЖАККАРА (родинність словників):")
for _, row in jaccard_df.iterrows():
    type_name = {'verbal': 'ДІЄСЛІВНІ', 'nominal': 'ІМЕННІ', 'adverbial': 'ПРИСЛІВНИКОВІ'}[row['type']]
    print(f"   {type_name}: {row['jaccard_index']:.3f} ({row['overlap_pct']:.1f}% перекриття)")
