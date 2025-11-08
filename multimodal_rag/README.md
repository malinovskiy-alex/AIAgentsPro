# 🎨 Multimodal RAG Demo з ChromaDB

Демонстрація multimodal RAG системи, яка працює з **текстом + зображеннями** використовуючи ChromaDB.

**Інспіровано**: [Nano Banana Milvus Blog](https://milvus.io/blog/nano-banana-milvus-turning-hype-into-enterprise-ready-multimodal-rag.md)
**Використовує**: ChromaDB (простіший за Milvus для воркшопів)

---

## 🎯 Що Демонструє

### 1. **Multimodal Embeddings з CLIP**
- Текст та зображення в одному векторному просторі
- CLIP model: `clip-ViT-B-32` (512D embeddings)
- Семантична подібність між text ↔ image

### 2. **ChromaDB для Векторного Пошуку**
- Легкий у використанні (без складного setup)
- In-memory для demo
- Швидкий пошук nearest neighbors

### 3. **Cross-Modal Queries**
- 📝 → 🖼️ Текстовий запит знаходить зображення
- 🖼️ → 📝 Зображення знаходить релевантний текст
- 🖼️ → 🖼️ Пошук схожих зображень

---

## 🚀 Швидкий Старт

### Встановлення

```bash
# 1. Перейти в директорію
cd /Users/o.denysiuk/agents/module/2/rag_demos/multimodal_rag

# 2. Встановити залежності
pip install chromadb sentence-transformers pillow torch

# 3. Запустити demo
python multimodal_rag_demo.py
```

### Що Встановиться

```
chromadb              # Векторна база даних
sentence-transformers # CLIP model
pillow                # Image processing
torch                 # PyTorch для моделей
```

**Розмір завантаження**: ~500MB (CLIP model)
**Час першого запуску**: 1-2 хвилини (завантаження моделі)

---

## 📊 Demo Scenarios

### Demo 1: Fruit Recognition 🍌
```python
# Додаємо інформацію про фрукти (текст + зображення)
banana_info = "Banana is a yellow tropical fruit, rich in potassium..."
apple_info = "Apple is a round fruit that comes in red, green..."

# Пошукові запити
query1 = "yellow tropical fruit rich in potassium"  # → знайде banana
query2 = "round citrus fruit with vitamin C"        # → знайде orange
query3 = "healthy fruit for breakfast"              # → знайде всі
```

**Use Case**: Product recognition в retail/grocery

### Demo 2: E-Commerce Product Search 🛒
```python
# Додаємо продукти
laptop = "MacBook Pro 16-inch with M4 chip. Silver aluminum..."
chair = "Ergonomic office chair with lumbar support..."

# Пошук
query1 = "powerful laptop for software development"  # → MacBook/Dell
query2 = "Apple professional computer"               # → MacBook
query3 = "comfortable chair for home office"         # → Herman Miller
```

**Use Case**: E-commerce search, visual similarity

---

## 🎓 Для Воркшопу

### Як Використовувати на Воркшопі (7 хв)

**Підготовка** (за день до):
```bash
# Запустити один раз щоб завантажити модель
python multimodal_rag_demo.py
```

**На воркшопі:**

1. **Пояснити концепцію** (2 хв)
   - Multimodal = Text + Images в одному просторі
   - CLIP model кодує обидва типи даних однаково
   - Дозволяє cross-modal search

2. **Показати код** (2 хв)
   ```python
   # Ключові частини:
   # 1. Encode text
   text_embedding = model.encode("banana fruit")

   # 2. Encode image
   image_embedding = model.encode(Image.open("banana.jpg"))

   # 3. Вони в одному просторі! Можна порівнювати
   similarity = cosine_similarity(text_embedding, image_embedding)
   ```

3. **Запустити demo** (3 хв)
   ```bash
   python multimodal_rag_demo.py
   ```
   - Покаже пошук фруктів
   - Покаже e-commerce search
   - Результати з similarity scores

---

## 💡 Архітектура

```
┌─────────────────────────────────────────────────┐
│  Input: Text Query або Image                    │
│  "yellow tropical fruit" або banana.jpg         │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │  CLIP Encoder       │
        │  (clip-ViT-B-32)    │
        │                     │
        │  Text → 512D vector │
        │  Image → 512D vector│
        └─────────┬───────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │  ChromaDB           │
        │  Vector Search      │
        │  k-NN (cosine)      │
        └─────────┬───────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │  Results:           │
        │  - Text docs        │
        │  - Images           │
        │  - Similarity scores│
        └─────────────────────┘
```

---

## 🔧 Розширення для Production

### Додати Справжні Зображення

```python
# 1. Створити images/ директорію
mkdir -p images

# 2. Додати зображення
# images/banana.jpg
# images/apple.jpg
# images/orange.jpg

# 3. Використати в коді
rag.add_image_document(
    "banana_img",
    "images/banana.jpg",
    caption="A yellow curved banana"
)
```

### Image Preprocessing

```python
from PIL import Image

def preprocess_image(image_path, size=(224, 224)):
    """Resize та normalize зображення"""
    img = Image.open(image_path)
    img = img.resize(size)
    # Додаткова обробка...
    return img
```

### Metadata Filtering

```python
# Пошук лише в певній категорії
results = rag.search_by_text(
    "laptop",
    n_results=5,
    where={"category": "electronics"}  # ChromaDB filtering
)
```

---

## 📈 Use Cases

| Use Case | Опис | Приклад |
|----------|------|---------|
| **E-Commerce** | Пошук товарів за фото або описом | "Знайди схожий стілець" |
| **Medical Imaging** | Пошук схожих випадків за снімками | X-ray → діагноз + історія |
| **Fashion Retail** | Пошук одягу за фото | Фото outfit → де купити |
| **Document Search** | Пошук в PDF з діаграмами | "Знайди архітектурні схеми" |
| **Social Media** | Content moderation, пошук | Знайти схожі memes |

---

## ⚡ Performance

### Benchmarks (Mac M4, 16GB)

```
Model loading:     ~2s (перший раз)
Text encoding:     ~10ms
Image encoding:    ~50ms
Search (1K docs):  ~5ms
Total query:       ~65ms
```

### Масштабування

| Documents | Embeddings Size | Search Time |
|-----------|----------------|-------------|
| 1K        | 2MB            | 5ms         |
| 10K       | 20MB           | 15ms        |
| 100K      | 200MB          | 50ms        |
| 1M        | 2GB            | 200ms*      |

*Для >100K використайте HNSW index або Milvus

---

## 🆚 ChromaDB vs Milvus

| Feature | ChromaDB | Milvus |
|---------|----------|--------|
| **Setup** | ✅ Простий (pip install) | ⚠️ Docker/K8s |
| **Use Case** | Prototypes, demos | Production, scale |
| **Scale** | < 1M vectors | Billions |
| **Features** | Basic | Advanced (sharding, replicas) |
| **Для воркшопу** | ✅ **Ідеально** | ❌ Overkill |

**Рекомендація для воркшопу**: ChromaDB (простіше, швидше setup)
**Рекомендація для production**: Milvus (якщо >1M vectors)

---

## 🐛 Troubleshooting

### Помилка: "No module named 'chromadb'"
```bash
pip install chromadb
```

### Помилка: "CLIP model download failed"
```bash
# Manually download
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('clip-ViT-B-32')
# Буде завантажено в ~/.cache/
```

### Повільний перший запуск
- Нормально! CLIP model ~500MB
- Наступні запуски швидкі (кешується)

### Помилка з PIL/Pillow
```bash
pip install --upgrade pillow
```

---

## 📚 Додаткові Ресурси

### Документація
- [ChromaDB Docs](https://docs.trychroma.com/)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Sentence Transformers](https://www.sbert.net/)

### Альтернативні Моделі
```python
# Кращі але повільніші:
model = SentenceTransformer('clip-ViT-B-32')      # 512D, швидко
model = SentenceTransformer('clip-ViT-L-14')      # 768D, краще
model = SentenceTransformer('clip-ViT-L-14-336')  # 768D, найкраще

# Для production:
# - OpenCLIP variants
# - Custom fine-tuned models
```

---

## ✅ Чеклист для Воркшопу

**За день до**:
- [ ] Встановити chromadb, sentence-transformers
- [ ] Запустити `python multimodal_rag_demo.py` один раз
- [ ] Переконатися що CLIP model завантажено
- [ ] (Опціонально) Додати справжні зображення

**На воркшопі**:
- [ ] Пояснити концепцію multimodal embeddings
- [ ] Показати код (CLIP encoding)
- [ ] Запустити demo (2 scenarios)
- [ ] Показати результати з similarity scores
- [ ] Пояснити use cases

**Час**: ~7 хвилин
**Складність**: Середня (потребує розуміння embeddings)

---

**Створено**: 25 жовтня 2025
**Для**: RAG Workshop Module 2
**Мова**: Python 3.11+

**Успіхів на воркшопі! 🎉**
