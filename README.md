# architecture-rag-llm

Решения по заданиям — в [Project_template.md](Project_template.md).

## База знаний

Источник — Rick and Morty Wiki. В `knowledge_base/` лежат 36 статей с вымышленными
названиями; один файл соответствует одной сущности. База на английском языке.
Источники и условия использования — в [DATA_SOURCES.md](DATA_SOURCES.md).

Подготовка данных (Python 3.11+):

```bash
python -m pip install -r requirements-data.txt
python prepare_knowledge_base.py --offline
python prepare_knowledge_base.py --check
python -m unittest discover -s tests -p 'test_prepare_knowledge_base.py' -v
```

`--offline` собирает базу из сохранённого снимка без сети. `--check` только
проверяет данные и их соответствие снимку, не меняя файлы.

Для скачивания текущих версий выбранных страниц через публичный MediaWiki API:

```bash
python prepare_knowledge_base.py
```

Ответы API кэшируются в `.cache/wiki/`. Для обновления конкретной статьи удаляется
её файл кэша; для полного обновления — каталог `.cache/wiki/`. Изменения вики могут
изменить результат. Точный повтор текущей версии делается через `--offline`.

`data/kb_sources.json` задаёт страницы и новые имена, `terms_map.json` — замены,
`data/source_snapshot.json` — очищенные исходные абзацы и сведения об источниках.
В RAG-индекс загружаются **только `.md` и `.txt` из `knowledge_base/`**.

## Векторный индекс

Модель — [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base),
768 измерений. Индекс — FAISS `IndexFlatIP`, векторы нормализованы для cosine similarity.
Версия модели закреплена в `vector_index/manifest.json`.

| Параметр | Значение |
| --- | --- |
| База | 36 документов из `knowledge_base/` |
| Чанки | 120 |
| Размер | 100–261 слово; один короткий документ сохранён целиком: 97 слов |
| Максимальный вход E5 | 409 токенов, включая префикс и заголовок |
| Генерация эмбеддингов | 41,652 с на CPU, 4 потока, batch size 8 |
| Сборка с загрузкой модели из кэша | 47,464 с; первое скачивание модели в замер не входит |

Установка и первая сборка (модель один раз скачивается в игнорируемую `.cache/models/`):

```bash
python -m pip install -r requirements-index.txt
python build_index.py
```

Повторная сборка без сети, поиск и проверка:

```bash
python build_index.py --offline
python search_index.py "Какую долю энергии Nezuko Delomi забирает из Lesser Realm?" --top-k 3
python evaluate_retrieval.py
python -m unittest discover -s tests -v
```

Поиск использует только локальный кэш модели. Сохраняются `vector_index/faiss.index`,
`chunks.jsonl` и `manifest.json`: тексты, ID, файл-источник, заголовок, разделы, строки,
символьные позиции и контрольные суммы. При изменении, добавлении или удалении
документов поиск требует пересборки индекса.

Для шести контрольных вопросов на русском и английском ожидаемый источник найден первым,
а нужные факты присутствуют в top-5. Полные запросы и найденные фрагменты:
[reports/retrieval_examples.md](reports/retrieval_examples.md) и JSON рядом.
Это небольшая проверка поиска, а не оценка качества ответов LLM. Score — cosine similarity,
не вероятность правильного ответа. Генерация ответов будет следующим этапом.
