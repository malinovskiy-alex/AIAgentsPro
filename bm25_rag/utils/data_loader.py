"""
Утиліти для завантаження та підготовки даних
"""
import os
import json
from typing import List, Dict
from pathlib import Path

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


class DocumentLoader:
    """Завантажувач документів для RAG систем"""

    def __init__(self, data_dir: str = "data/corporate_docs"):
        self.data_dir = Path(data_dir)

    def load_documents(self, max_documents: int = None) -> List[Dict[str, str]]:
        """
        Завантажує всі текстові та PDF документи з директорії

        Args:
            max_documents: Максимальна кількість документів для завантаження (None = всі)

        Returns:
            List[Dict]: Список документів з метаданими
        """
        documents = []

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Директорія {self.data_dir} не існує")

        # Спочатку збираємо всі файли
        all_files = list(self.data_dir.glob("*.txt")) + list(self.data_dir.glob("*.pdf"))

        # Обмежуємо кількість якщо потрібно
        if max_documents:
            all_files = all_files[:max_documents]

        for file_path in all_files:
            try:
                if file_path.suffix == '.txt':
                    # Читаємо текстові файли
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                elif file_path.suffix == '.pdf':
                    # Читаємо PDF файли
                    if not HAS_PYMUPDF:
                        print(f"⚠️  PyMuPDF не встановлено, пропускаємо {file_path.name}")
                        continue

                    doc = fitz.open(file_path)
                    content = ""
                    for page in doc:
                        content += page.get_text()
                    doc.close()
                else:
                    continue

                documents.append({
                    "content": content,
                    "source": file_path.name,
                    "path": str(file_path)
                })
            except Exception as e:
                print(f"⚠️  Помилка читання {file_path.name}: {e}")
                continue

        print(f"✅ Завантажено {len(documents)} документів")
        return documents

    def load_test_queries(self, queries_file: str = "data/test_queries.json") -> Dict:
        """
        Завантажує тестові запити

        Args:
            queries_file: Шлях до JSON файлу з запитами

        Returns:
            Dict: Словник з категоріями запитів
        """
        queries_path = Path(queries_file)

        if not queries_path.exists():
            raise FileNotFoundError(f"Файл {queries_file} не існує")

        with open(queries_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)

        # Підтримка як dict так і list формату
        if isinstance(queries, list):
            # Конвертувати list в dict
            queries_dict = {"general": queries}
            total = len(queries)
        else:
            queries_dict = queries
            total = sum(len(v) for v in queries.values())

        print(f"✅ Завантажено {total} тестових запитів")
        return queries_dict

    def load_unified_queries(
        self,
        queries_file: str = "data/test_queries_unified.json",
        max_queries: int = None,
        categories: List[str] = None
    ) -> List[Dict]:
        """
        Завантажує уніфікований тестовий датасет (100 запитів)

        Це єдиний датасет для всіх RAG підходів - дозволяє коректне порівняння!

        Args:
            queries_file: Шлях до уніфікованого JSON файлу
            max_queries: Максимальна кількість запитів (None = всі)
            categories: Список категорій для фільтрації (None = всі)

        Returns:
            List[Dict]: Список тестових запитів
        """
        queries_path = Path(queries_file)

        if not queries_path.exists():
            raise FileNotFoundError(
                f"Уніфікований датасет не знайдено: {queries_file}\n"
                f"Цей файл містить 100 стандартизованих запитів для порівняння RAG підходів."
            )

        with open(queries_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        queries = data.get("queries", [])

        # Фільтрація по категоріях
        if categories:
            queries = [q for q in queries if q.get("category") in categories]

        # Обмеження кількості
        if max_queries:
            queries = queries[:max_queries]

        print(f"✅ Завантажено уніфікований датасет: {len(queries)} запитів")
        if categories:
            print(f"   Категорії: {', '.join(categories)}")

        return queries


class TextSplitter:
    """Розбиття тексту на чанки"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Розбиває текст на чанки з перекриттям

        Args:
            text: Вхідний текст

        Returns:
            List[str]: Список чанків
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]

            # Намагаємось розбити по реченнях
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                split_point = max(last_period, last_newline)

                if split_point > self.chunk_size * 0.5:
                    chunk = text[start:start + split_point + 1]
                    end = start + split_point + 1

            chunks.append(chunk.strip())
            start = end - self.chunk_overlap

        return chunks

    def split_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Розбиває документи на чанки зі збереженням метаданих

        Args:
            documents: Список документів

        Returns:
            List[Dict]: Список чанків з метаданими
        """
        all_chunks = []

        for doc in documents:
            chunks = self.split_text(doc["content"])

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "content": chunk,
                    "source": doc["source"],
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                })

        print(f"✅ Створено {len(all_chunks)} чанків з {len(documents)} документів")
        return all_chunks


def save_results(results: Dict, output_file: str):
    """
    Зберігає результати оцінки у файл

    Args:
        results: Словник з результатами
        output_file: Шлях до вихідного файлу
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Результати збережено в {output_file}")


def print_results(results: Dict):
    """
    Виводить результати у читабельному форматі

    Args:
        results: Словник з результатами
    """
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТИ ОЦІНКИ")
    print("="*60)

    if "system_name" in results:
        print(f"\n🔧 Система: {results['system_name']}")

    if "metrics" in results:
        print(f"\n📈 Метрики:")
        for metric, value in results["metrics"].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")

    if "execution_time" in results:
        print(f"\n⏱️  Час виконання: {results['execution_time']:.2f} секунд")

    print("\n" + "="*60)
