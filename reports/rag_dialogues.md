# Примеры работы RAG-бота

Модель: Qwen3.5-9B Q5_K_M; llama.cpp b12058; thinking выключен.
Каждый пример — независимый вопрос. Ответы получены от локальной модели, без ручного редактирования.
Проверка: статус, ожидаемый источник и контрольные слова. Это smoke-тесты, а не полная оценка достоверности.
Пройдено: 7/7.

Условия запуска: CPU: Ryzen 7 7700, 8 vCPU в виртуальной машине, 16 ГБ RAM; llama.cpp использует 4 потока. GPU недоступна. Контекст 4096, top-k 3, temperature 0, seed 42.

## fuel

**Пользователь:** Какая субстанция служит топливом для межпространственного Rift Weaving?

**Бот:**

Проверка по источникам:
1. Veyl Catalyst выполняет функцию топлива для межпространственного Rift Weaving. [S1]
   «Because of these properties, Veyl Catalyst functions as both the fuel and essential medium for Veyran's interdimensional Rift Weaving, rather than simply serving as a conventional energy source.»

Ответ: Veyl Catalyst служит топливом для межпространственного Rift Weaving. [S1]

Источники:
[S1] Veyl Catalyst — knowledge_base/veyl_catalyst.md:1-19; veyl_catalyst:000:581d22d76955

Время: 126.729 с; токенов промпта: 1650.

## energy

**Пользователь:** Какую долю энергии Nezuko Delomi забирает из Lesser Realm?

**Бот:**

Проверка по источникам:
1. Nezuko Delomi забирает 80% энергии, производимой жителями его Lesser Realm. [S1]
   «Nezuko Delomi states that 80% of the energy made by the inhabitants of his Lesser Realm is extracted for his use»

Ответ: Nezuko Delomi забирает 80% энергии из Lesser Realm. [S1]

Источники:
[S1] Nested Realm Cell — knowledge_base/nested_realm_cell.md:11-13; nested_realm_cell:001:307f5b5904bb

Время: 104.489 с; токенов промпта: 1612.

## government

**Пользователь:** Какой вид основал правительство Vhalan Compact и управлял им?

**Бот:**

Проверка по источникам:
1. Правительство Vhalan Compact было основано и управлялось Yavari. [S1]
   «The Compact government was founded and ruled by the bug-like Yavari, although it includes members of many other species.»

Ответ: Правительство Vhalan Compact основал и управлял им Yavari. [S1]

Источники:
[S1] Vhalan Compact — knowledge_base/vhalan_compact.md:1-15; vhalan_compact:000:6445ebe43427

Время: 102.345 с; токенов промпта: 1564.

## cyborg

**Пользователь:** Какой киборг был создан из Avor Kel?

**Бот:**

Проверка по источникам:
1. Avor Kel был воскрешен как киборг под именем Avor Ember. [S1]
   «Despite being shot multiple times, Avor Kel is soon resurrected by the Compact as the cyborg Avor Ember in order to serve them.»
2. Sera и Yavari возродили Avor Kel в виде киборга и переименовали его в Avor Ember. [S2]
   «Sera and the Yavari resurrected Avor Kel as a cyborg and renamed him Avor Ember.»

Ответ: Из Avor Kel был создан киборг Avor Ember. [S1] [S2]

Источники:
[S1] Avor Kel — knowledge_base/avor_kel.md:1-9; avor_kel:000:a48ad1f75126
[S2] Avor Kel — knowledge_base/avor_kel.md:17-25; avor_kel:002:b8fad729ef06

Время: 130.823 с; токенов промпта: 1730.

## entrance

**Пользователь:** Как попасть в Velvet Expanse и какие свойства среды защищали детей от падения и утопления?

**Бот:**

Проверка по источникам:
1. Вход в Velvet Expanse осуществляется путем рисования входа с помощью светящегося фиолетового мелка. [S1]
   «Velvet Expanse is accessed by drawing an entranceway with what appears to be a glowing purple piece of chalk.»
2. Поверхности земли пружинистые, чтобы предотвратить травмы от падения, а вода в реке радуги дышащая, чтобы предотвратить утопление. [S1]
   «All ground surfaces are bouncy to prevent injuries from falling, the rainbow river's water is breathable to prevent drowning»

Ответ: Чтобы попасть в Velvet Expanse, нужно нарисовать вход светящимся фиолетовым мелком. Защита от падения обеспечивается пружинистыми поверхностями, а от утопления — дышащей водой в реке радуги. [S1]

Источники:
[S1] Velvet Expanse — knowledge_base/velvet_expanse.md:9-15; velvet_expanse:001:0c1bf47fccdc

Время: 266.353 с; токенов промпта: 1662.

## unknown_volume

**Пользователь:** Каков точный объём топливного бака Rift Spindle в литрах?

**Бот:**

Я не знаю.

Время: 227.649 с; токенов промпта: 1483.

## unknown_birthday

**Пользователь:** Назови точную дату рождения Avor Kel: день, месяц и год.

**Бот:**

Я не знаю.

Время: 156.586 с; токенов промпта: 1735.
