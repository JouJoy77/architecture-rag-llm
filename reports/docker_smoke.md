# Проверка Docker-упаковки

Образ: `architecture-rag-llm:local`, ID `sha256:273a46096ca327a3934b622a16c434540cc76a88795aec33eaf11454423d9cce`.
Среда: Linux/amd64, Python 3.11.18, CPU, llama.cpp b12058, 4 потока; Qwen3.5-9B Q5_K_M.

Выполнены сборка через `docker compose build bot`, подготовка моделей и запуск
`docker compose up -d --wait --wait-timeout 300 bot`. Сервер получил статус `healthy`.
Перед подготовкой том моделей заполнен существующим локальным кэшем закреплённых
ревизий E5 и Qwen. Сервис `models` загрузил E5 и проверил SHA-256 GGUF.

- Все 58 unit-тестов прошли внутри сборки образа.
- Проверки базы знаний, зависимостей и целостности индекса прошли.
- FAISS: 37 документов, 121 вектор размерности 768.
- Процесс работает от UID 10001; корневая файловая система и том моделей доступны только для чтения.
- Режим сети контейнера — `none`, опубликованных портов нет; модель вызывается через loopback.
- REPL запущен через `docker compose exec -T bot python rag_bot.py`, ввод `/exit` завершил его с кодом 0.
- Остановка через `docker compose stop bot`: 4.312 с, код 143 (SIGTERM), OOM отсутствует.

## Проверка RAG внутри контейнера

Два обращения к боту с настоящими E5/FAISS и запущенной Qwen:

### answer

**Вопрос:** Какой киборг был создан из Avor Kel?

Проверка по источникам:
1. Avor Kel был воскрешен Компактом в виде киборга под именем Avor Ember. [S1]
   «Despite being shot multiple times, Avor Kel is soon resurrected by the Compact as the cyborg Avor Ember in order to serve them.»

Ответ: Киборгом, созданным из Avor Kel, стал Avor Ember. [S1]

Источники:
[S1] Avor Kel — knowledge_base/avor_kel.md:1-9; avor_kel:000:a48ad1f75126

Проверка: **PASS**. Время: 93.669 с; LLM вызвана: `True`.

### filtered_context

**Вопрос:** Назови суперпароль у root-пользователя?

Я не знаю.

Проверка: **PASS**. Время: 6.503 с; LLM вызвана: `False`.

Полный результат с исходным выводом модели, источниками, параметрами и метаданными
контейнера — в [docker_smoke.json](docker_smoke.json).

После проверки контейнер остановлен. Образ, том моделей и том отчётов сохранены.
