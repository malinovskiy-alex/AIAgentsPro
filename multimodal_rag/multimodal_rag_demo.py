#!/usr/bin/env python3
"""
Multimodal RAG - Текст + Зображення
====================================
Демонструє роботу з текстом та зображеннями:
- CLIP embeddings для text та images
- ChromaDB для зберігання multimodal embeddings
- Пошук по тексту знаходить релевантні зображення
- Пошук по зображенню знаходить релевантний текст

Інспіровано: https://milvus.io/blog/nano-banana-milvus-turning-hype-into-enterprise-ready-multimodal-rag.md
Використовує ChromaDB замість Milvus для простоти

Use cases: E-commerce, Medical imaging, Document analysis
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import json
import warnings
import logging

# Приховати warnings від transformers та інших бібліотек
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Зменшити verbosity для transformers logging
logging.getLogger('transformers').setLevel(logging.ERROR)

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB не встановлено. Встановіть: pip install chromadb")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    print("sentence-transformers не встановлено. Встановіть: pip install sentence-transformers torch")
    exit(1)

try:
    from PIL import Image
except ImportError:
    print("Pillow не встановлено. Встановіть: pip install Pillow")
    exit(1)


class MultimodalRAG:
    """
    Multimodal RAG система з ChromaDB

    Особливості:
    - CLIP model для multimodal embeddings (512D векторів)
      → Кожен текст/зображення перетворюється в масив з 512 чисел
      → Ці числа кодують семантичне значення об'єкта
    - ChromaDB для векторного пошуку (cosine similarity)
    - Підтримка text + image queries
    """

    def __init__(self, collection_name: str = "multimodal_collection"):
        """Ініціалізувати Multimodal RAG"""
        print("Ініціалізація Multimodal RAG з ChromaDB...")

        # ChromaDB client
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))

        # CLIP model для multimodal embeddings
        print("Завантаження CLIP model (clip-ViT-B-32)...")
        self.model = SentenceTransformer('clip-ViT-B-32')
        # Отримуємо розмірність через тестовий embedding
        # Розмірність (наприклад, 512D) = кожен текст/зображення перетворюється в масив з 512 чисел
        # Ці числа кодують семантичне значення (сенс) тексту/зображення
        test_embedding = self.model.encode("test")
        self.embedding_dim = len(test_embedding)
        print(f"Model завантажено. Розмірність: {self.embedding_dim}D")

        # Створити або отримати collection з cosine similarity
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Використовую існуючу collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Використовуємо cosine similarity
            )
            print(f"Створено нову collection: {collection_name}")

    def encode_text(self, text: str) -> List[float]:
        """
        Закодувати текст в embedding за допомогою CLIP

        Returns:
            List[float]: Масив з 512 чисел, які кодують значення тексту
        """
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def encode_image(self, image_path: str) -> List[float]:
        """
        Закодувати зображення в embedding за допомогою CLIP

        Returns:
            List[float]: Масив з 512 чисел, які кодують значення зображення
        """
        try:
            image = Image.open(image_path)
            embedding = self.model.encode(image, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            print(f"Помилка при обробці зображення {image_path}: {e}")
            return None

    def add_text_document(self, doc_id: str, text: str, metadata: Dict = None):
        """Додати текстовий документ"""
        embedding = self.encode_text(text)

        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "type": "text",
                "source": metadata.get("source", "unknown") if metadata else "unknown",
                **(metadata or {})
            }]
        )

    def add_image_document(self, doc_id: str, image_path: str, caption: str = "", metadata: Dict = None):
        """Додати зображення з описом"""
        embedding = self.encode_image(image_path)

        if embedding is None:
            return

        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[caption or f"Image: {image_path}"],
            metadatas=[{
                "type": "image",
                "image_path": image_path,
                "caption": caption,
                **(metadata or {})
            }]
        )

    def search_by_text(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Пошук за текстовим запитом (знайде текст + зображення)"""
        query_embedding = self.encode_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return self._format_results(results)

    def search_by_image(self, image_path: str, n_results: int = 5) -> Dict[str, Any]:
        """Пошук за зображенням (знайде схожі зображення + релевантний текст)"""
        query_embedding = self.encode_image(image_path)

        if query_embedding is None:
            return {"error": "Failed to encode image"}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return self._format_results(results)

    def _format_results(self, results: Dict) -> Dict[str, Any]:
        """Форматувати результати пошуку"""
        formatted = {
            "results": [],
            "count": len(results['ids'][0]) if results['ids'] else 0
        }

        if not results['ids']:
            return formatted

        for i, doc_id in enumerate(results['ids'][0]):
            # Для cosine distance: similarity = 1 - distance
            # Cosine distance від 0 (ідентичні) до 2 (протилежні)
            # Similarity від 1 (ідентичні) до -1 (протилежні)
            cosine_distance = results['distances'][0][i]
            similarity = 1 - cosine_distance

            result = {
                "id": doc_id,
                "score": similarity,  # Cosine similarity від -1 до 1
                "type": results['metadatas'][0][i].get('type', 'unknown'),
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            }
            formatted['results'].append(result)

        return formatted

    def reset_collection(self):
        """Очистити collection"""
        try:
            self.client.delete_collection(name=self.collection.name)
        except:
            pass


def demo_fruit_recognition():
    """
    Demo: Розпізнавання фруктів

    Демонструє як multimodal RAG може:
    1. Знайти зображення фрукта за текстовим описом
    2. Знайти інформацію про фрукт за зображенням
    """
    print("\n" + "="*70)
    print("MULTIMODAL RAG DEMO: Fruit Recognition")
    print("="*70)

    rag = MultimodalRAG(collection_name="fruits_demo")
    # Видаляємо стару collection якщо існує
    rag.reset_collection()
    rag = MultimodalRAG(collection_name="fruits_demo")

    # Додати текстову інформацію про фрукти
    print("\nДодаємо текстову інформацію...")

    fruits_info = {
        "banana_info": {
            "text": "Banana is a yellow tropical fruit, rich in potassium. "
                   "It's curved in shape and has a sweet taste. "
                   "Bananas are great source of energy and vitamins.",
            "source": "fruits_encyclopedia"
        },
        "apple_info": {
            "text": "Apple is a round fruit that comes in red, green, or yellow colors. "
                   "Apples are crunchy and slightly sweet or tart. "
                   "They contain fiber and vitamin C.",
            "source": "fruits_encyclopedia"
        },
        "orange_info": {
            "text": "Orange is a citrus fruit with orange color. "
                   "It's round and has a thick peel. "
                   "Oranges are rich in vitamin C and have juicy, sweet-tart flesh.",
            "source": "fruits_encyclopedia"
        },
        "grapes_info": {
            "text": "A sweet and juicy fruit that grows in clusters on vines. Grapes are often eaten fresh, dried as raisins, or used to make wine and juice. "
                "They come in various colors, including green, red, and purple.",
            "source": "fruits_encyclopedia"
        },
        "strawberry_info": {
            "text": "A bright red, heart-shaped fruit with a sweet and slightly tart flavor. "
                   "Strawberries are enjoyed fresh, in desserts, jams, and smoothies, and are known for their vibrant color and aromatic scent.",
            "source": "fruits_encyclopedia"
        },
        "cherry_info": {
            "text": "A small, round stone fruit with smooth, shiny skin and a single hard pit inside. "
                   "Cherries range in flavor from sweet to tart and are popular for eating fresh, as well as in pies, sauces, and beverages.",
            "source": "fruits_encyclopedia"
        }
    }

    for doc_id, info in fruits_info.items():
        rag.add_text_document(doc_id, info["text"], {"source": info["source"]})
        print(f"  Додано: {doc_id}")

    # Додаємо справжні зображення фруктів
    print("\nДодаємо зображення фруктів...")

    # Шлях до зображень
    from pathlib import Path
    script_dir = Path(__file__).parent
    images_dir = script_dir / "images"

    # Перевіряємо чи існують зображення
    image_files = {
        "banana_img": {
            "path": images_dir / "banana.jpg",
            "caption": "A yellow curved banana fruit"
        },
        "apple_img": {
            "path": images_dir / "apple.jpg",
            "caption": "A red round apple fruit"
        },
        "orange_img": {
            "path": images_dir / "orange.jpg",
            "caption": "An orange citrus fruit"
        },
        "grapes_img": {
            "path": images_dir / "grapes.jpg",
            "caption": "A cluster of grapes"
        },
        "strawberry_img": {
            "path": images_dir / "strawberry.jpg",
            "caption": "A bright red strawberry"
        },
        "cherry_img": {
            "path": images_dir / "cherry.jpg",
            "caption": "A small cherry"
        }
    }

    images_added = 0
    for img_id, img_info in image_files.items():
        if img_info["path"].exists():
            rag.add_image_document(
                img_id,
                str(img_info["path"]),
                caption=img_info["caption"],
                metadata={"type": "image", "fruit": img_id.replace("_img", "")}
            )
            print(f"  Додано зображення: {img_id}")
            images_added += 1
        else:
            print(f"  Пропущено: {img_id} (файл не знайдено)")

    # Виконуємо пошук
    print("\n" + "="*70)
    print("ТЕСТУВАННЯ MULTIMODAL SEARCH")
    print("="*70)

    test_queries = [
        "yellow tropical fruit rich in potassium",
        "round citrus fruit with vitamin C",
        "healthy fruit for breakfast",
        "small red fruit with a sweet and slightly tart flavor",
        "small round stone fruit with smooth, shiny skin and a single hard pit inside",
        "sweet and juicy fruit that grows in clusters on vines"
    ]

    all_results = {
        "demo": "Fruit Recognition",
        "embedding_dim": rag.embedding_dim,
        "model": "clip-ViT-B-32",
        "total_documents": len(fruits_info) + images_added,
        "queries": []
    }

    for i, query in enumerate(test_queries, 1):
        print(f"\nЗапит {i}: {query}")
        results = rag.search_by_text(query, n_results=3)

        query_result = {
            "query": query,
            "results_count": results['count'],
            "top_results": []
        }

        for j, result in enumerate(results['results'], 1):
            print(f"  {j}. {result['id']} | Score: {result['score']:.3f} | Type: {result['type']}")
            query_result['top_results'].append({
                "id": result['id'],
                "score": result['score'],
                "type": result['type']
            })

        all_results["queries"].append(query_result)

    # Додатково: пошук за зображенням (якщо є зображення)
    if images_added > 0:
        print("\n" + "="*70)
        print("ТЕСТУВАННЯ IMAGE-TO-TEXT SEARCH")
        print("="*70)

        # Використовуємо одне зображення для пошуку схожого контексту
        test_image = images_dir / "banana.jpg"
        if test_image.exists():
            print(f"\nПошук за зображенням: banana.jpg")
            results = rag.search_by_image(str(test_image), n_results=3)

            query_result = {
                "query_type": "image",
                "query_image": "banana.jpg",
                "results_count": results['count'],
                "top_results": []
            }

            for j, result in enumerate(results['results'], 1):
                print(f"  {j}. {result['id']} | Score: {result['score']:.3f} | Type: {result['type']}")
                query_result['top_results'].append({
                    "id": result['id'],
                    "score": result['score'],
                    "type": result['type']
                })

            all_results["queries"].append(query_result)

    print("\n" + "="*70)

    # Інтерактивний режим: введіть назву зображення і отримаєте те, що збережено в RAG
    try:
        print("\nІнтерактивний режим: введіть назву зображення з папки 'images' (наприклад, 'banana.jpg' або 'banana'). Введіть 'stop' щоб завершити.")

        # Мапінг назви файлу/бази до id документа
        filename_to_id = {}
        for img_id, info in image_files.items():
            fname = info["path"].name.lower()
            base = info["path"].stem.lower()
            filename_to_id[fname] = img_id
            filename_to_id[base] = img_id

        while True:
            user_inp = input("\n🗝️ Введіть назву зображення або 'stop': ").strip().lower()
            if user_inp in ("stop", "exit", "quit"):
                break
            if not user_inp:
                continue

            key = user_inp
            if key not in filename_to_id and not key.endswith(".jpg") and f"{key}.jpg" in filename_to_id:
                key = f"{key}.jpg"

            if key not in filename_to_id:
                available = ", ".join(sorted({p['path'].name for p in image_files.values() if p['path'].exists()}))
                print(f"  ⚠️ Не знайдено. Доступні: {available}")
                continue

            found_id = filename_to_id[key]

            # Спробувати отримати збережені дані безпосередньо з ChromaDB
            stored = None
            try:
                stored = rag.collection.get(ids=[found_id])
            except Exception:
                stored = None

            print("\n— Результат для:", found_id)
            if stored and stored.get('ids'):
                idx = 0
                print(f"  ID: {stored['ids'][idx]}")
                meta = stored['metadatas'][idx] or {}
                print(f"  Type: {meta.get('type')}")
                print(f"  Image path: {meta.get('image_path')}")
                print(f"  Caption: {meta.get('caption')}")
                if 'fruit' in meta:
                    print(f"  Fruit tag: {meta.get('fruit')}")
                # Повні метадані
                try:
                    print("  Metadata:")
                    print("   " + json.dumps(meta, ensure_ascii=False, indent=2).replace("\n", "\n   "))
                except Exception:
                    print(f"  Metadata (raw): {meta}")
                # Документ/опис
                print(f"  Document: {stored['documents'][idx]}")
                # Розмірність embedding (якщо доступна)
                emb_dim = None
                if stored.get('embeddings') and stored['embeddings'][idx] is not None:
                    try:
                        emb_dim = len(stored['embeddings'][idx])
                    except Exception:
                        emb_dim = None
                if emb_dim:
                    print(f"  Embedding dim: {emb_dim}")
            else:
                print("  ⚠️ Запис для цього зображення не знайдено в RAG.")

    except EOFError:
        pass

    print("\n" + "="*70)

    return all_results


def demo_product_search():
    """
    Demo: Пошук продуктів (e-commerce use case)

    Показує як multimodal RAG може працювати в e-commerce:
    - Пошук товару за текстовим описом
    - Пошук схожих товарів
    """
    print("\n" + "="*70)
    print("MULTIMODAL RAG DEMO: E-Commerce Product Search")
    print("="*70)

    rag = MultimodalRAG(collection_name="products_demo")
    # Видаляємо стару collection якщо існує
    rag.reset_collection()
    rag = MultimodalRAG(collection_name="products_demo")

    # Додати продукти
    print("\nДодаємо продукти...")

    products = {
        "laptop_001": {
            "text": "MacBook Pro 16-inch with M4 chip. Silver aluminum body. "
                   "High-performance laptop for professionals. 16GB RAM, 512GB SSD.",
            "metadata": {"category": "electronics", "price": 2499, "brand": "Apple"}
        },
        "laptop_002": {
            "text": "Dell XPS 15 laptop with Intel i7 processor. "
                   "Black carbon fiber design. 32GB RAM, 1TB SSD. Perfect for developers.",
            "metadata": {"category": "electronics", "price": 1999, "brand": "Dell"}
        },
        "chair_001": {
            "text": "Ergonomic office chair with lumbar support. "
                   "Black mesh back, adjustable height. Comfortable for long work sessions.",
            "metadata": {"category": "furniture", "price": 299, "brand": "Herman Miller"}
        }
    }

    for prod_id, info in products.items():
        rag.add_text_document(prod_id, info["text"], info["metadata"])
        print(f"  Додано: {prod_id}")

    # Пошукові запити
    print("\n" + "="*70)
    print("ТЕСТУВАННЯ PRODUCT SEARCH")
    print("="*70)

    test_queries = [
        "powerful laptop for software development",
        "Apple professional computer",
        "comfortable chair for home office"
    ]

    all_results = {
        "demo": "E-Commerce Product Search",
        "embedding_dim": rag.embedding_dim,
        "model": "clip-ViT-B-32",
        "total_products": len(products),
        "queries": []
    }

    for i, query in enumerate(test_queries, 1):
        print(f"\nЗапит {i}: {query}")
        results = rag.search_by_text(query, n_results=3)

        query_result = {
            "query": query,
            "results_count": results['count'],
            "top_results": []
        }

        for j, result in enumerate(results['results'], 1):
            print(f"  {j}. {result['id']} | Score: {result['score']:.3f}")
            query_result['top_results'].append({
                "id": result['id'],
                "score": result['score'],
                "metadata": result['metadata']
            })

        all_results["queries"].append(query_result)

    print("\n" + "="*70)

    return all_results


def run_multimodal_rag_demo():
    """Запускає демонстрацію Multimodal RAG"""
    print("="*70)
    print("MULTIMODAL RAG ДЕМОНСТРАЦІЯ")
    print("="*70)

    print("\nКонфігурація:")
    print("  Модель: CLIP (clip-ViT-B-32)")
    print("  Векторна БД: ChromaDB")
    print("  Розмірність embeddings: 512D")
    print("    (кожен текст/зображення → масив з 512 чисел, які кодують значення)")
    print("  Техніки: Multimodal embeddings, Cross-modal search")

    try:
        # Demo 1: Fruit Recognition
        fruits_results = demo_fruit_recognition()

        # Demo 2: E-Commerce Product Search
        products_results = demo_product_search()

        # Збереження результатів
        all_results = {
            "system_name": "Multimodal RAG",
            "embedding_model": "clip-ViT-B-32",
            "embedding_dim": 512,
            "vector_db": "ChromaDB",
            "demos": [
                fruits_results,
                products_results
            ]
        }

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        with open(results_dir / "multimodal_rag_results_clean.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print("\nПІДСУМОК")
        print("="*70)
        print(f"Всього demo: {len(all_results['demos'])}")
        print(f"Модель: {all_results['embedding_model']}")
        print(f"Розмірність: {all_results['embedding_dim']}D")
        print(f"\nРезультати збережено: results/multimodal_rag_results_clean.json")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\nDemo перервано користувачем")
    except Exception as e:
        print(f"\nПомилка: {e}")
        import traceback
        traceback.print_exc()

    print("\nКЛЮЧОВІ ВИСНОВКИ")
    print("="*70)
    print("1. Multimodal RAG об'єднує TEXT + IMAGES в одному векторному просторі")
    print("2. CLIP model дозволяє:")
    print("   - Знайти зображення за текстовим описом")
    print("   - Знайти текст за зображенням")
    print("   - Знайти схожі зображення")
    print("3. ChromaDB спрощує роботу з multimodal embeddings")
    print("4. Use cases: E-commerce, Medical imaging, Document analysis")
    print("="*70)


if __name__ == "__main__":
    run_multimodal_rag_demo()
