# Bitcoin Data Collector

Скрипт для сбора данных об адресах и транзакциях Bitcoin с сервиса WalletExplorer.com через JSON API.

## Структура проекта

```
bitcoin_data_collector/
├── data/                           # CSV файлы с данными
│   ├── addresses.csv              # Данные об адресах
│   ├── addresses-labels.csv       # Метки адресов
│   ├── transactions.csv           # Данные о транзакциях
│   └── bitcoin_address_features_*.csv  # Результаты анализа
├── bitcoin_collector.py           # Скрипт сбора данных
├── bitcoin_transaction_analyzer.py # Анализатор транзакций
├── monitor.py                     # Мониторинг прогресса
└── requirements.txt               # Зависимости
```

## Установка

```bash
pip install -r requirements.txt
```

## Использование

```bash
python bitcoin_collector.py input.csv
```

### Параметры

- `input_csv` - путь к CSV файлу с адресами (первый столбец должен содержать адреса)
- `--delay` - задержка между запросами в секундах (по умолчанию: 1.0)
- `--addresses-output` - имя файла для сохранения данных об адресах (по умолчанию: addresses.csv)
- `--transactions-output` - имя файла для сохранения данных о транзакциях (по умолчанию: transactions.csv)
- `--jsonl-output` - имя JSONL файла для сохранения сырых ответов API (по умолчанию: raw_responses.jsonl)
- `--parallel` - использовать параллельную обработку (ускоряет в 3-5 раз)
- `--max-workers` - максимальное количество потоков для параллельной обработки (по умолчанию: 5)

### Примеры

**Обычный режим:**
```bash
python bitcoin_collector.py example_input.csv --delay 0.5
```

**Параллельный режим (рекомендуется):**
```bash
python bitcoin_collector.py example_input.csv --parallel --max-workers 10 --delay 0.1
```

**Максимальная скорость:**
```bash
python bitcoin_collector.py example_input.csv --parallel --max-workers 20 --delay 0.05
```

## Выходные файлы

### raw_responses.jsonl
Содержит все сырые JSON ответы от API в формате JSON Lines:
- `type` - тип запроса (address_info, transaction_details)
- `address`/`txid` - идентификатор запроса
- `url` - URL запроса
- `params` - параметры запроса
- `response` - полный JSON ответ от API
- `timestamp` - время запроса

### addresses.csv
Содержит данные об адресах:
- address - Bitcoin адрес
- wallet_name - название кошелька
- wallet_id - ID кошелька
- found - найден ли адрес в базе
- txs_count - количество транзакций
- updated_to_block - последний обновленный блок

### transactions.csv
Содержит данные о транзакциях:
- address - Bitcoin адрес
- txid - ID транзакции
- amount_sent - отправленная сумма
- amount_received - полученная сумма
- block_height - высота блока
- block_pos - позиция в блоке
- time - время транзакции
- balance - баланс после транзакции
- used_as_input - использован как вход
- used_as_output - использован как выход
- found - найдена ли транзакция
- label - метка кошелька
- wallet_id - ID кошелька
- size - размер транзакции
- is_coinbase - является ли coinbase транзакцией
- updated_to_block - последний обновленный блок
- inputs_count - количество входов
- outputs_count - количество выходов
- inputs - массив входов (JSON)
- outputs - массив выходов (JSON)

## Инкрементальный сбор данных

Скрипт автоматически определяет уже обработанные адреса и транзакции, что позволяет:

- **Продолжать сбор** с места остановки
- **Избегать дублирования** данных
- **Ускорить повторные запуски** в разы
- **Безопасно прерывать** и возобновлять процесс

При запуске скрипт:
1. Загружает уже обработанные адреса из `addresses.csv`
2. Загружает уже обработанные транзакции из `transactions.csv`
3. Обрабатывает только новые данные
4. Показывает статистику: "Найдено X адресов, из них Y новых для обработки"

## Анализ транзакций

После сбора данных можно запустить анализ для извлечения признаков:

```bash
python bitcoin_transaction_analyzer.py
```

### Мониторинг прогресса

```bash
python monitor.py
```

## Логирование

Скрипт создает файл `bitcoin_analyzer.log` с подробной информацией о процессе анализа.
