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
python -m unittest discover -s tests -v
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
В будущий RAG-индекс загружается **только `knowledge_base/*.md`**.
