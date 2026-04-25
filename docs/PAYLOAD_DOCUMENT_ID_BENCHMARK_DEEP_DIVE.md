# Подробная Спецификация Payload, `document_id` И Benchmark В OmniGuard 2.0

Этот документ предназначен не для поверхностного знакомства, а для детального технического разбора.

Его цель:

1. подробно объяснить, как устроен payload;
2. показать, как именно используется `document_id`;
3. разобрать, как реально работает benchmark;
4. выписать все формулы, которые используются в коде;
5. дать примеры, которые можно использовать в дипломе.

---

## 1. Где в коде реализована эта логика

Основные файлы:

- `omniguard/payload.py` — упаковка, кодирование, декодирование и проверка payload
- `omniguard/legacy_models.py` — встраивание и извлечение payload через pretrained-модель
- `omniguard/service.py` — объединение payload-анализа, heatmap и comparison-метрик
- `omniguard/benchmark.py` — логика benchmark
- `omniguard/metrics.py` — формулы метрик
- `omniguard/attacks.py` — атаки, используемые в benchmark

---

## 2. Общая идея payload

Payload — это компактная полезная нагрузка, которая скрыто встраивается в изображение.

Важно понимать разницу:

- **tamper-sensitive watermark** нужен для локализации возможной правки;
- **payload** нужен для хранения и проверки компактных данных, связанных с изображением.

В текущей реализации payload отвечает на три ключевых вопроса:

1. какой `document_id` был связан с изображением;
2. когда был сформирован payload;
3. сохранилась ли встроенная информация после искажений или правок.

---

## 3. Формальный формат payload

### 3.1. Константы формата

В `omniguard/payload.py` заданы:

- `PAYLOAD_VERSION = 1`
- `PAYLOAD_RAW_BITS = 56`
- `PAYLOAD_CODE_BITS = 98`
- `PAYLOAD_PAD_BITS = 2`
- `PAYLOAD_TOTAL_BITS = 100`
- `PAYLOAD_EPOCH = 2026-01-01 00:00:00 UTC`

Идея такая:

- у нас есть **56 полезных бит**;
- они кодируются Hamming(7,4);
- после кодирования получается **98 бит**;
- затем добавляются **2 добивочных бита**, чтобы уложиться ровно в **100 бит**, которые умеет переносить payload-ветка pretrained-модели.

### 3.2. Разбиение raw payload по полям

56 полезных бит распределяются так:

1. `4 бита` — версия формата
2. `20 бит` — время в часах относительно `PAYLOAD_EPOCH`
3. `16 бит` — hash от `document_id`
4. `8 бит` — `nonce`
5. `8 бит` — HMAC-тег

То есть:

`56 = 4 + 20 + 16 + 8 + 8`

---

## 4. Как используется `document_id`

`document_id` — это строковый идентификатор изображения или документа.

Примеры:

- `diploma-demo-001`
- `report-2026-04-26`
- `passport-scan-17`

### 4.1. Почему `document_id` не хранится в открытом виде

Модель может перенести только 100 бит. Поэтому внутри изображения нельзя хранить длинную строку в явном виде.

Поэтому используется компромисс:

- строка `document_id` переводится в hash;
- из hash берется компактное представление в 16 бит;
- уже эти 16 бит записываются в payload.

### 4.2. Формула получения hash от `document_id`

Реализация:

```python
digest = hashlib.sha256(document_id.encode("utf-8")).digest()
value = int.from_bytes(digest[:8], "big") & ((1 << width) - 1)
```

При `width = 16` это означает:

1. берется `SHA-256(document_id)`;
2. берутся первые 8 байт digest;
3. они интерпретируются как одно 64-битное число;
4. затем берутся младшие 16 бит.

Формально:

Пусть:

- `D = SHA256(document_id)`
- `X = int(D[0:8])`

Тогда:

`document_hash_16 = X mod 2^16`

После этого число переводится в 16-битный двоичный вектор.

### 4.3. Зачем это нужно

Позже, при декодировании, система делает то же самое с ожидаемым `document_id` и сравнивает:

`decoded_document_hash_bits == expected_document_hash_bits`

Если совпало:

- `payload.document_match = True`

Если нет:

- `payload.document_match = False`

То есть `document_id` используется как компактный идентификатор принадлежности изображения.

---

## 5. Как кодируются числа в биты

В payload используются две базовые функции:

### 5.1. `_int_to_bits(value, width)`

Функция переводит целое число в двоичное представление фиксированной длины.

Формально:

Если задано число `v` и ширина `w`, то функция возвращает:

`bits(v, w) = двоичная запись v длины w`

Например:

- `bits(5, 4) = 0101`
- `bits(17, 8) = 00010001`

### 5.2. `_bits_to_int(bits)`

Выполняет обратную операцию:

`int(bits) = число, соответствующее двоичной записи`

Например:

- `int(0101) = 5`
- `int(00010001) = 17`

---

## 6. Как кодируется время в payload

Время не хранится в формате ISO-строки. Оно хранится как число часов от фиксированной эпохи.

### 6.1. Формула

Пусть:

- `T` — текущее время в UTC;
- `E` — `PAYLOAD_EPOCH = 2026-01-01 00:00:00 UTC`

Тогда:

`issued_hours = floor((T - E) / 3600 секунд)`

В коде:

```python
issued_hours = int((issued_at_utc - PAYLOAD_EPOCH).total_seconds() // 3600)
```

### 6.2. Почему именно 20 бит

20 бит позволяют хранить значения от `0` до `2^20 - 1 = 1,048,575`.

Это значит, что можно покрыть:

`1,048,576 часов ≈ 119.7 лет`

То есть формата хватает на очень большой временной диапазон.

---

## 7. `nonce`: что это такое и зачем нужен

`nonce` — это случайное 8-битное число.

В коде:

```python
nonce = secrets.randbelow(2**8)
```

Формально:

`nonce ∈ [0, 255]`

Зачем он нужен:

1. чтобы payload не был полностью детерминированным только по `document_id`;
2. чтобы одинаковый `document_id` мог давать разные итоговые битовые последовательности в разное время или в разных сессиях;
3. чтобы уменьшить предсказуемость payload.

---

## 8. HMAC-тег в payload

После того как сформированы первые 48 бит:

- версия
- время
- hash `document_id`
- `nonce`

система вычисляет HMAC-тег.

### 8.1. Формула

Пусть:

- `B` — строка бит base payload длины 48;
- `K` — секретный ключ `hmac_secret`

Тогда:

`H = HMAC_SHA256(K, ascii(B))`

В коде:

```python
message = "".join(str(bit) for bit in bits).encode("ascii")
digest = hmac.new(secret_key.encode("utf-8"), message, hashlib.sha256).digest()
value = digest[0] & ((1 << width) - 1)
```

При `width = 8`:

`auth_tag_8 = digest[0] mod 2^8`

Так как `digest[0]` уже байт, фактически берется 8-битное значение первого байта HMAC-digest.

### 8.2. Почему тег только 8 бит

Потому что payload сильно ограничен по вместимости.

Это важное инженерное ограничение:

- полный HMAC-SHA256 занимает 256 бит;
- payload-канал вмещает только 100 бит;
- после размещения версии, времени, hash и `nonce` остается только 8 бит на контрольную аутентификацию.

То есть это **не полноценная криптографическая подпись в сильном смысле**, а компактный контрольный тег в условиях жесткого бюджетного ограничения по битам.

---

## 9. Формирование raw payload

После вычисления всех полей raw payload строится конкатенацией:

`raw_bits = version_bits || issued_hours_bits || document_hash_bits || nonce_bits || auth_bits`

Где:

- `||` означает конкатенацию.

Формально:

`raw_bits ∈ {0,1}^56`

---

## 10. Hamming(7,4): кодирование payload

После формирования 56 raw bits используется код Хэмминга `(7,4)`.

### 10.1. Зачем нужен Hamming-код

Он позволяет:

- кодировать каждые 4 полезных бита в 7-битное кодовое слово;
- исправлять одиночную битовую ошибку в каждом 7-битном слове.

Это важно для watermark-сценария, потому что после атак или сжатия некоторые биты могут извлекаться с ошибками.

### 10.2. Как делится поток

56 бит делятся на группы по 4:

`56 / 4 = 14 групп`

Каждая группа превращается в 7 бит:

`14 * 7 = 98 бит`

После этого добавляются 2 padding-бита:

`98 + 2 = 100 бит`

### 10.3. Формула кодирования одного nibble

Пусть входные биты:

- `d1, d2, d3, d4`

Тогда проверочные биты:

- `p1 = d1 XOR d2 XOR d4`
- `p2 = d1 XOR d3 XOR d4`
- `p3 = d2 XOR d3 XOR d4`

Итоговое 7-битное слово:

`[p1, p2, d1, p3, d2, d3, d4]`

### 10.4. Пример кодирования одного nibble

Пусть:

- `d1=1`
- `d2=0`
- `d3=1`
- `d4=1`

Тогда:

- `p1 = 1 XOR 0 XOR 1 = 0`
- `p2 = 1 XOR 1 XOR 1 = 1`
- `p3 = 0 XOR 1 XOR 1 = 0`

Итог:

`[0, 1, 1, 0, 0, 1, 1]`

То есть:

`1011 -> 0110011`

### 10.5. Padding

После 98 кодовых бит код добавляет:

`[0, 1]`

Итоговый поток:

`encoded_bits = hamming_code(raw_bits) || [0, 1]`

---

## 11. Декодирование Hamming(7,4)

При декодировании каждые 7 бит рассматриваются как кодовое слово.

### 11.1. Формулы синдрома

Пусть кодовое слово:

`bits = [b1, b2, b3, b4, b5, b6, b7]`

В коде:

```python
s1 = bits[0] ^ bits[2] ^ bits[4] ^ bits[6]
s2 = bits[1] ^ bits[2] ^ bits[5] ^ bits[6]
s3 = bits[3] ^ bits[4] ^ bits[5] ^ bits[6]
syndrome = s1 + (s2 << 1) + (s3 << 2)
```

Формально:

- `s1 = b1 XOR b3 XOR b5 XOR b7`
- `s2 = b2 XOR b3 XOR b6 XOR b7`
- `s3 = b4 XOR b5 XOR b6 XOR b7`

И:

`syndrome = s1 + 2*s2 + 4*s3`

### 11.2. Исправление ошибки

Если `syndrome != 0`, то:

`error_index = syndrome - 1`

и соответствующий бит инвертируется.

Это и есть исправление одиночной ошибки.

### 11.3. Извлечение полезных бит

После коррекции полезные биты берутся из позиций:

- `b3`
- `b5`
- `b6`
- `b7`

То есть:

`decoded_nibble = [b3, b5, b6, b7]`

---

## 12. Итоговый процесс формирования payload

Полная цепочка такая:

1. взять `document_id`
2. посчитать `SHA-256(document_id)`
3. выделить 16 бит hash
4. вычислить `issued_hours`
5. сгенерировать `nonce`
6. собрать 48 base bits
7. вычислить 8 HMAC bits
8. получить 56 raw bits
9. закодировать Hamming(7,4) → 98 bits
10. добавить 2 pad bits
11. получить 100 final bits

---

## 13. Реальные примеры payload

Ниже приведены реальные примеры, полученные из текущего кода.

### Пример 1

Вход:

- `document_id = "report-2026-04-26"`
- `issued_at_utc = 2026-04-26T12:00:00+00:00`
- `nonce = 17`
- `hmac_secret = "omniguard-demo-key"`

Получено:

- `document_hash_hex = 1bbf`
- `auth_tag_hex = b0`

Raw bits:

```text
00010000000010101101010000011011101111110001000110110000
```

Encoded bits:

```text
1101001000000000000001011010101010110011001101001011001101100111111111110100111010010110011000000001
```

### Пример 2

Вход:

- `document_id = "diploma-demo-001"`
- `issued_at_utc = 2026-01-15T08:00:00+00:00`
- `nonce = 203`
- `hmac_secret = "omniguard-demo-key"`

Получено:

- `document_hash_hex = f2a4`
- `auth_tag_hex = 86`

Raw bits:

```text
00010000000000010101100011110010101001001100101110000110
```

Encoded bits:

```text
1101001000000000000001101001010010111100001111111010101010110101001100011110001100111110000110011001
```

### 13.1. Что показывают эти примеры

Они демонстрируют:

1. одно и то же устройство payload работает детерминированно при фиксированных времени, `nonce` и ключе;
2. разные `document_id` дают разные hash bits;
3. разные входные данные приводят к полностью различным 100-битным потокам.

---

## 14. Как payload встраивается в изображение

Эта логика находится в `omniguard/legacy_models.py`, в методе `embed_payload_bits(...)`.

### 14.1. Проверка длины payload

Сначала код проверяет, что встраиваемый поток имеет длину ровно 100 бит:

`len(bits) == payload_bit_length`

где `payload_bit_length = 100`.

### 14.2. Изображение для payload-ветки

Исходное изображение приводится к разрешению:

`payload_resolution = 256`

То есть перед входом в payload-энкодер выполняется:

`payload_image = resize(source, 256 x 256)`

### 14.3. Нормализация

Изображение переводится из диапазона `[0, 255]` в `[0, 1]`, а затем для payload-энкодера — в `[-1, 1]`.

Формально:

`x01 = image / 255`

`x11 = 2 * x01 - 1`

### 14.4. Работа payload encoder

В коде:

```python
stego_small = self.model.bm.encoder(source_small * 2.0 - 1.0, secret)
```

Где:

- `source_small` — изображение `256x256`
- `secret` — тензор из 100 бит

### 14.5. Residual-схема

После работы encoder берется не просто финальное маленькое изображение, а его отличие от исходного:

`residual_small = stego_small - source_small`

Затем residual интерполируется до полного размера:

`residual_full = interpolate(residual_small, full_size)`

И добавляется к полному изображению:

`stego_full = clamp(source_full + strength * residual_full, 0, 1)`

### 14.6. Зачем это сделано

Это значит, что payload встраивается через **масштабируемый residual**, а не через полную замену изображения.

Именно поэтому параметр:

`payload_strength`

контролирует интенсивность встраивания.

---

## 15. Как payload извлекается

Метод:

`LegacyModelBundle.decode_payload_bits(...)`

### 15.1. Шаги

1. изображение приводится к `256x256`
2. нормализуется в `[-1, 1]`
3. прогоняется через `self.model.bm.decoder`
4. логиты threshold-ятся по нулю

Формула:

Если `logit_i > 0`, то:

`bit_i = 1`

иначе:

`bit_i = 0`

---

## 16. Как payload проверяется при анализе

В `OmniGuardEngine.analyze_image(...)` делается:

1. `predicted_bits = self.models.decode_payload_bits(source)`
2. `payload_result = decode_payload_bits(...)`

### 16.1. Что делает `decode_payload_bits(...)`

1. декодирует Hamming-код;
2. восстанавливает:
   - version
   - issued_hours
   - document_hash_bits
   - nonce
   - auth_bits
3. вычисляет ожидаемый `auth_bits` повторно;
4. сравнивает:

`decoded_auth_bits == recomputed_auth_bits`

Если равны:

`auth_ok = True`

### 16.2. Как проверяется `document_id`

Если пользователь передал `expected_document_id`, то:

`document_match = decoded_document_hash_bits == hash_bits(expected_document_id, 16)`

То есть сравнивается не строка, а 16-битный hash этой строки.

---

## 17. Важные ограничения payload

### 17.1. `document_id` хранится в hash-виде

Значит:

- это не точная строка;
- это компактный идентификатор принадлежности.

### 17.2. HMAC короткий

Всего 8 бит — это мало для сильной криптографической гарантии, но это осознанный компромисс в рамках 100-битного канала.

### 17.3. `bit_accuracy` и `decoded_bits`

В текущей реализации `PayloadDecodeResult.decoded_bits` сохраняет:

`encoded_bits[:100]`

То есть benchmark сравнивает:

- исходные 100 encoded bits
- восстановленные 100 encoded bits

Именно поэтому:

`payload_bit_accuracy`

в benchmark отражает точность восстановления **финального 100-битного потока**, а не только 56 raw bits.

Это важно правильно объяснять.

---

## 18. Что такое benchmark в OmniGuard

Benchmark — это автоматический стенд оценки устойчивости системы.

Он отвечает на вопрос:

> Что произойдет с watermark, payload и локализацией, если защищенное изображение подвергнуть различным атакам?

То есть benchmark проверяет не только визуальное качество, но и:

1. устойчивость payload;
2. устойчивость tamper-detection;
3. качество локализации правки.

---

## 19. Полный pipeline benchmark

Реализация находится в `omniguard/benchmark.py`, класс `BenchmarkRunner`.

### 19.1. Шаг 1. Создание защищенной версии

Сначала benchmark всегда создает защищенную версию:

`protection = engine.protect_image(image, document_id)`

То есть benchmark тестирует не исходное изображение, а именно защищенное.

### 19.2. Шаг 2. Сохранение protected image

Сохраняются:

- `protected.png`
- `protected.json`

### 19.3. Шаг 3. Применение атак

Для каждой атаки из `DEFAULT_ATTACKS` создается атакованное изображение.

### 19.4. Шаг 4. Анализ атакованного изображения

Для каждой атаки выполняется:

```python
analysis = self.engine.analyze_image(
    attacked.image,
    expected_document_id=document_id,
    reference_bits=protection.payload.encoded_bits,
    output_dir=attack_dir,
    reference_image=protected,
    analysis_mode="hybrid",
)
```

Это очень важная строчка.

Она означает, что benchmark анализирует атакованное изображение:

1. зная ожидаемый `document_id`
2. зная эталонные encoded bits payload
3. зная защищенную reference-версию
4. используя `hybrid` режим локализации

То есть benchmark работает в самом сильном диагностическом режиме из доступных в системе.

### 19.5. Шаг 5. Подсчет метрик

После анализа считаются image-метрики, payload-метрики и mask-метрики.

### 19.6. Шаг 6. Формирование JSON и CSV

В конце benchmark сохраняет:

- `benchmark_report.json`
- `benchmark_report.csv`

---

## 20. Набор атак benchmark

По умолчанию используются:

1. `identity`
2. `jpeg_q70`
3. `gaussian_blur`
4. `resize_065`
5. `brightness_115`
6. `copy_move`
7. `rect_inpaint`

Ниже — что именно делает каждая.

### 20.1. `identity`

Формально:

`I_attacked = I_protected`

Это контрольный тест.

Зачем нужен:

- чтобы понять baseline системы без искажения.

### 20.2. `jpeg_q70`

Изображение кодируется в JPEG с качеством `70`, затем декодируется обратно.

Формально:

`I_attacked = JPEG_decode(JPEG_encode(I_protected, q=70))`

Зачем:

- проверить устойчивость к типичному сжатию с потерями.

### 20.3. `gaussian_blur`

Используется размытие Гаусса с радиусом `1.5`.

Формально:

`I_attacked = G_sigma(I_protected)`

где `sigma` соответствует blur-радиусу `1.5`.

### 20.4. `resize_065`

Изображение уменьшается в `0.65` раза, затем растягивается обратно.

Формально:

`I_small = resize(I_protected, 0.65)`

`I_attacked = resize(I_small, original_size)`

### 20.5. `brightness_115`

Меняется яркость с коэффициентом `1.15`.

Формально:

`I_attacked = clamp(1.15 * I_protected, 0, 255)`

### 20.6. `copy_move`

Из изображения вырезается patch и копируется в другое место.

Алгоритм:

1. выбирается прямоугольный patch;
2. patch копируется в новую позицию;
3. формируется ground-truth mask для области вставки.

Размеры patch:

- `box_h = max(16, height // 6)`
- `box_w = max(16, width // 6)`

Это локальная структурная атака.

### 20.7. `rect_inpaint`

Создается прямоугольная маска и участок заполняется через OpenCV inpaint.

Размеры patch:

- `box_h = max(24, height // 5)`
- `box_w = max(24, width // 5)`

Это уже атака, имитирующая удаление объекта или локальную ретушь.

---

## 21. Как benchmark получает heatmap и бинарную маску

Все это делает `OmniGuardEngine.analyze_image(...)`.

### 21.1. Watermark map

Сначала модель строит сырую карту аномалии:

`watermark_raw = self.models.reveal_tamper_mask(source, scales)`

Затем карта нормализуется по percentiles:

`watermark_map = normalize(watermark_raw, p_low=80.0, p_high=99.8)`

Смысл нормализации:

Если:

- `low = percentile(score, 80)`
- `high = percentile(score, 99.8)`

то:

`normalized = clip((score - low) / (high - low), 0, 1)`

Это подавляет фон и усиливает высокие значения.

### 21.2. Reference map

Если есть защищенная опорная версия, строится карта абсолютных различий.

Пусть:

- `S = source / 255`
- `R = reference / 255`

Тогда:

`channel_diff = abs(S - R)`

Далее:

- берется grayscale-версия этого отличия
- берется `max_channel` по RGB
- затем:

`blended = max(gray_diff, max_channel_diff)`

После этого опять выполняется percentile-нормализация:

`reference_map = normalize(blended, p_low=65.0, p_high=99.5)`

### 21.3. Гибридная карта

Если выбран режим `hybrid`, то:

`combined = clip(0.35 * watermark_map + 0.65 * reference_map, 0, 1)`

Затем:

`combined = max(combined, 0.95 * reference_map)`

После этого:

`combined = GaussianBlur(combined, sigma=1.1)`

### 21.4. Только watermark

Если режим `watermark`:

`combined = watermark_map`

### 21.5. Только reference

Если режим `reference`:

`combined = reference_map`

---

## 22. Как строится бинарная маска

### 22.1. Без reference

Если `reference_map` нет:

`binary_mask = 255 * [heatmap >= threshold]`

Где:

- `threshold` — пользовательский порог бинарной маски.

### 22.2. С reference

Если `reference_map` есть, маска строится более строго.

Сначала берется Otsu threshold:

`otsu_value = OTSU(reference_map_uint8)`

Затем:

`reference_threshold = max(user_threshold, otsu_value / 255)`

Промежуточные маски:

`direct_mask = [reference_map >= max(0.08, 0.7 * reference_threshold)]`

`combined_mask = [heatmap >= max(user_threshold, 0.85 * reference_threshold)]`

Финальная маска:

`binary = direct_mask AND (combined_mask OR [reference_map >= 0.22])`

После этого применяется морфология:

1. opening kernel `3x3`
2. closing kernel `7x7`
3. dilate kernel `3x3`

Это нужно, чтобы:

- убрать мелкий шум;
- закрыть дыры в маске;
- сделать контуры более цельными.

---

## 23. Формулы benchmark-метрик

### 23.1. MSE

Пусть:

- `A` — защищенное изображение
- `B` — атакованное изображение
- `N` — число всех пиксельных значений

Тогда:

`MSE(A, B) = (1 / N) * Σ (A_i - B_i)^2`

### 23.2. MAE

`MAE(A, B) = (1 / N) * Σ |A_i - B_i|`

### 23.3. RMSE

`RMSE(A, B) = sqrt(MSE(A, B))`

### 23.4. PSNR

Для 8-битных изображений:

`PSNR(A, B) = 10 * log10((255^2) / MSE(A, B))`

Если `MSE <= 1e-10`, в коде возвращается:

`PSNR = 100`

### 23.5. SSIM

В коде используется `skimage.metrics.structural_similarity`.

Концептуально SSIM сравнивает:

- яркость
- контраст
- структуру

Общая форма:

`SSIM(x, y) = l(x, y) * c(x, y) * s(x, y)`

где:

- `l` — компонент яркости
- `c` — компонент контраста
- `s` — компонент структуры

### 23.6. Payload Bit Accuracy

В benchmark:

`payload_bit_accuracy = correct_bits / compared_bits`

где сравниваются:

- исходный `protection.payload.encoded_bits`
- восстановленный `analysis.payload.decoded_bits`

Формально:

`BitAccuracy = (1 / M) * Σ [b_i == b'_i]`

### 23.7. Payload Auth OK

Это булев флаг:

`auth_ok = (decoded_auth_bits == recomputed_auth_bits)`

### 23.8. Payload Document Match

Булев флаг:

`document_match = (decoded_document_hash_bits == hash_bits(expected_document_id, 16))`

### 23.9. Tamper Score Mean

Пусть `H` — итоговая heatmap.

Тогда:

`TamperScoreMean = mean(H)`

### 23.10. Tamper Score Max

`TamperScoreMax = max(H)`

### 23.11. Tamper Ratio

Пусть `M_bin` — бинарная маска.

После нормализации в `[0,1]`:

`TamperRatio = mean(M_bin / 255)`

То есть это просто доля пикселей, вошедших в итоговую маску.

### 23.12. Precision

Пусть:

- `TP` — true positives
- `FP` — false positives

Тогда:

`Precision = TP / (TP + FP)`

Если знаменатель равен нулю, код возвращает `1.0`.

### 23.13. Recall

Пусть:

- `TP` — true positives
- `FN` — false negatives

Тогда:

`Recall = TP / (TP + FN)`

Если знаменатель равен нулю, код возвращает `1.0`.

### 23.14. F1

`F1 = 2 * Precision * Recall / (Precision + Recall)`

Если знаменатель почти нулевой, код возвращает `0.0`.

### 23.15. IoU

Пусть:

- `A` — эталонная маска
- `B` — предсказанная маска

Тогда:

`IoU = |A ∩ B| / |A ∪ B|`

Если объединение пусто, код возвращает `1.0`.

### 23.16. Dice

`Dice = 2 * |A ∩ B| / (|A| + |B|)`

Если обе маски пусты, код возвращает `1.0`.

### 23.17. Changed Pixel Ratio

Для двух изображений:

1. считается максимальное отличие по каналам:

`diff(x, y) = max_c |A(x, y, c) - B(x, y, c)|`

2. пиксель считается измененным, если:

`diff(x, y) >= threshold`

В коде по умолчанию:

`threshold = 12`

3. итоговая метрика:

`ChangedPixelRatio = changed_pixels / total_pixels`

---

## 24. Когда какие mask-метрики считаются в benchmark

Если атака не возвращает ground-truth mask:

- `mask_precision`
- `mask_recall`
- `mask_f1`
- `mask_iou`
- `mask_dice`

не считаются.

Это относится, например, к:

- `identity`
- `jpeg_q70`
- `gaussian_blur`
- `resize_065`
- `brightness_115`

А вот для:

- `copy_move`
- `rect_inpaint`

эталонная маска есть, поэтому эти метрики вычисляются.

---

## 25. Что именно benchmark сохраняет на диск

Для каждого запуска:

1. `protected.png`
2. `protected.json`
3. для каждой атаки:
   - `attacked.png`
   - `attack.json`
   - при наличии — `ground_truth_mask.png`
   - `tamper_heatmap.png`
   - `tamper_mask.png`
4. `benchmark_report.json`
5. `benchmark_report.csv`

---

## 26. Что означает benchmark методологически

Benchmark в OmniGuard — это не просто “таблица чисел”.

Это методология из трех уровней оценки:

### Уровень 1. Визуальная устойчивость

Оценивается метриками:

- `MSE`
- `MAE`
- `RMSE`
- `PSNR`
- `SSIM`

### Уровень 2. Устойчивость встроенных данных

Оценивается метриками:

- `payload_bit_accuracy`
- `payload_auth_ok`
- `payload_document_match`

### Уровень 3. Качество локализации изменений

Оценивается метриками:

- `tamper_score_mean`
- `tamper_score_max`
- `tamper_ratio`
- `precision`
- `recall`
- `F1`
- `IoU`
- `Dice`

То есть benchmark одновременно отвечает на три вопроса:

1. насколько изменилась картинка визуально;
2. насколько пострадал payload;
3. насколько хорошо модель локализовала изменение.

---

## 27. Как это объяснять в дипломе

Если нужно кратко и технически корректно сформулировать:

> В системе payload представляет собой 56-битную полезную нагрузку, включающую версию формата, временную метку, укороченный hash `document_id`, случайный `nonce` и HMAC-тег. Далее payload кодируется кодом Хэмминга (7,4), что дает 98 кодовых бит, после чего добавляются 2 padding-бита для получения итогового 100-битного потока, встраиваемого pretrained-моделью в изображение. В benchmark-сценарии система сначала строит защищенную версию изображения, затем применяет набор искажений и локальных атак, после чего оценивает визуальную устойчивость, устойчивость payload и качество локализации изменений с помощью набора скалярных и mask-метрик.

---

## 28. Самые важные выводы

1. `document_id` в системе не хранится как строка, а как 16-битный hash.
2. Payload в реальности — это 56 raw bits, которые превращаются в 100 embedded bits.
3. Hamming(7,4) нужен для коррекции одиночных битовых ошибок.
4. HMAC-тег нужен для компактной проверки целостности payload.
5. Benchmark работает не на исходном изображении, а на его защищенной версии.
6. Benchmark оценивает не только картинку, но и payload, и локализацию изменений.
7. Для локальных атак самыми важными метриками обычно являются:
   - `payload_auth_ok`
   - `payload_document_match`
   - `tamper_score_max`
   - `tamper_ratio`
   - `IoU`
   - `Dice`

---

## 29. Что важно честно отметить как ограничения

1. 16-битный hash `document_id` может иметь коллизии.
2. 8-битный HMAC-тег является инженерным компромиссом, а не сильной криптографической подписью.
3. `payload_bit_accuracy` в benchmark оценивает восстановление именно encoded 100-битного потока.
4. Метрики локализации зависят от порога маски и режима локализации.
5. Качество benchmark зависит от выбранного набора атак и параметров этих атак.

---

## 30. Короткая формула-выжимка

Если нужен сверхкраткий формальный summary:

### Payload

`payload_raw = version(4) || time_hours(20) || hash(document_id)(16) || nonce(8) || hmac(8)`

`payload_encoded = Hamming(7,4)(payload_raw) || [0,1]`

### Проверка `document_id`

`document_match = (decoded_hash == hash(expected_document_id))`

### Benchmark

`image -> protect -> attacks -> analyze -> metrics -> report`

### Главные метрики

`PSNR = 10 * log10(255^2 / MSE)`

`IoU = intersection / union`

`Dice = 2 * intersection / (|A| + |B|)`

`F1 = 2PR / (P + R)`

`TamperRatio = suspicious_pixels / total_pixels`
