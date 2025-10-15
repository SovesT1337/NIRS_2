#!/usr/bin/env python3
"""
Bitcoin Transaction Analyzer

Анализирует транзакции Bitcoin и извлекает специфические признаки для классификации адресов.
Учитывает особенности Bitcoin: UTXO, входы/выходы, комиссии, временные паттерны и др.

Author: AI Assistant
License: MIT
"""

import json
import logging
import math
import statistics
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

# Configure logging
import os
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bitcoin_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BitcoinFeatures:
    """Структура для хранения извлеченных признаков Bitcoin адреса."""
    
    # Базовые признаки
    address: str = ""
    txs_count: int = 0
    label: str = ""
    
    # Структурные признаки транзакций
    avg_inputs: float = 0.0
    median_inputs: float = 0.0
    avg_outputs: float = 0.0
    median_outputs: float = 0.0
    max_inputs: int = 0
    max_outputs: int = 0
    min_inputs: int = 0
    min_outputs: int = 0
    inputs_outputs_ratio: float = 0.0
    avg_tx_size: float = 0.0
    median_tx_size: float = 0.0
    
    # UTXO паттерны
    avg_utxo_age: float = 0.0
    dust_ratio: float = 0.0  # доля транзакций < 546 satoshi
    change_output_ratio: float = 0.0
    largest_input_ratio: float = 0.0
    many_small_inputs_ratio: float = 0.0
    
    # Временные паттерны
    transaction_frequency: float = 0.0  # транзакций в день
    burst_activity_ratio: float = 0.0
    time_span_days: int = 0
    
    # Экономические паттерны
    avg_amount_sent: float = 0.0
    median_amount_sent: float = 0.0
    avg_amount_received: float = 0.0
    median_amount_received: float = 0.0
    max_single_tx: float = 0.0
    min_single_tx: float = 0.0
    round_number_ratio: float = 0.0
    value_entropy: float = 0.0
    
    # Сетевые паттерны
    unique_input_addresses: int = 0
    unique_output_addresses: int = 0
    address_reuse_ratio: float = 0.0
    
    # Bitcoin-специфичные
    coinbase_ratio: float = 0.0


class BitcoinTransactionAnalyzer:
    """Анализатор Bitcoin транзакций для извлечения признаков классификации."""
    
    # Константы
    DUST_THRESHOLD = 0.00000546  # 546 satoshi
    ROUND_AMOUNTS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    BURST_WINDOW_HOURS = 24
    MIN_BURST_TRANSACTIONS = 3
    SAVE_INTERVAL = 50  # Сохраняем каждые 50 адресов
    
    def __init__(self, dust_threshold: Optional[float] = None):
        """
        Инициализация анализатора.
        
        Args:
            dust_threshold: Порог для определения пылевых транзакций (в BTC)
        """
        self.dust_threshold = dust_threshold or self.DUST_THRESHOLD
        self.round_amounts = self.ROUND_AMOUNTS
        
    def load_data(
        self, 
        addresses_file: str, 
        transactions_file: str, 
        labels_file: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Загружает данные из CSV файлов.
        
        Args:
            addresses_file: Путь к файлу с адресами
            transactions_file: Путь к файлу с транзакциями
            labels_file: Путь к файлу с метками
            
        Returns:
            Tuple из трех DataFrame: адреса, транзакции, метки
            
        Raises:
            FileNotFoundError: Если файл не найден
            pd.errors.EmptyDataError: Если файл пустой
        """
        logger.info("Загрузка данных...")
        
        try:
            # Загружаем адреса
            addresses_df = pd.read_csv(addresses_file)
            logger.info(f"Загружено {len(addresses_df)} адресов")
            
            # Загружаем транзакции - только нужные колонки для скорости
            transactions_df = pd.read_csv(
                transactions_file,
                usecols=['address', 'txid', 'amount_sent', 'amount_received', 
                        'block_height', 'time', 'inputs_count', 'outputs_count', 
                        'size', 'is_coinbase'],
                dtype={'address': str, 'inputs_count': 'int32', 'outputs_count': 'int32'}
            )
            logger.info(f"Загружено {len(transactions_df)} транзакций")
            
            # Загружаем метки
            labels_df = pd.read_csv(labels_file)
            logger.info(f"Загружено {len(labels_df)} меток")
            
            return addresses_df, transactions_df, labels_df
            
        except FileNotFoundError as e:
            logger.error(f"Файл не найден: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            logger.error(f"Пустой файл: {e}")
            raise
    
    def _parse_json_field(self, json_str: str) -> List[Dict[str, Any]]:
        """
        Парсит JSON поле из CSV.
        
        Args:
            json_str: JSON строка для парсинга
            
        Returns:
            Список словарей с данными
        """
        if pd.isna(json_str) or json_str == '' or json_str == '[]':
            return []
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Ошибка парсинга JSON: {e}")
            return []
    
    def _calculate_utxo_age(
        self, 
        transaction: pd.Series, 
        all_transactions: pd.DataFrame
    ) -> float:
        """
        Вычисляет средний возраст UTXO для транзакции.
        
        Args:
            transaction: Серия с данными транзакции
            all_transactions: DataFrame со всеми транзакциями
            
        Returns:
            Средний возраст UTXO в секундах
        """
        if transaction['inputs_count'] == 0:
            return 0.0
            
        inputs = self._parse_json_field(transaction['inputs'])
        if not inputs:
            return 0.0
            
        ages = []
        for inp in inputs:
            if 'next_tx' in inp and inp['next_tx']:
                # Находим предыдущую транзакцию
                prev_tx = all_transactions[all_transactions['txid'] == inp['next_tx']]
                if not prev_tx.empty:
                    age = transaction['time'] - prev_tx.iloc[0]['time']
                    ages.append(age)
        
        return statistics.mean(ages) if ages else 0.0
    
    def _is_dust_transaction(self, amount: float) -> bool:
        """
        Проверяет, является ли транзакция пылевой.
        
        Args:
            amount: Сумма транзакции в BTC
            
        Returns:
            True если транзакция пылевая
        """
        return amount < self.dust_threshold
    
    def _is_round_number(self, amount: float) -> bool:
        """
        Проверяет, является ли сумма 'круглой'.
        
        Args:
            amount: Сумма для проверки
            
        Returns:
            True если сумма круглая
        """
        for round_amt in self.round_amounts:
            if abs(amount - round_amt) < 0.0001:
                return True
        return False
    
    def _calculate_entropy(self, amounts) -> float:
        """
        Вычисляет энтропию распределения сумм.
        
        Args:
            amounts: Список сумм
            
        Returns:
            Энтропия распределения
        """
        if amounts.size == 0:
            return 0.0
        
        try:
            # Создаем гистограмму
            bins = np.histogram(amounts, bins=10)[0]
            probabilities = bins / sum(bins)
            probabilities = probabilities[probabilities > 0]  # Убираем нули
            
            return -sum(p * math.log2(p) for p in probabilities)
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def _calculate_burst_activity(
        self, 
        timestamps: List[int], 
        window_hours: int = BURST_WINDOW_HOURS
    ) -> float:
        """
        Вычисляет долю транзакций в 'всплесках' активности.
        
        Args:
            timestamps: Список временных меток
            window_hours: Размер окна в часах
            
        Returns:
            Доля транзакций в всплесках
        """
        if len(timestamps) < 2:
            return 0.0
        
        timestamps = sorted(timestamps)
        burst_count = 0
        window_seconds = window_hours * 3600
        
        for i in range(len(timestamps)):
            # Считаем транзакции в окне
            window_start = timestamps[i]
            window_end = window_start + window_seconds
            window_txs = sum(1 for ts in timestamps if window_start <= ts <= window_end)
            
            if window_txs > self.MIN_BURST_TRANSACTIONS:
                burst_count += window_txs
        
        return burst_count / len(timestamps) if timestamps else 0.0
    
    
    def analyze_address_transactions(
        self, 
        address: str, 
        transactions: pd.DataFrame
    ) -> BitcoinFeatures:
        """
        Анализирует транзакции конкретного адреса и извлекает признаки.
        
        Args:
            address: Адрес для анализа
            transactions: DataFrame с транзакциями адреса
            all_transactions: DataFrame со всеми транзакциями
            
        Returns:
            Объект BitcoinFeatures с извлеченными признаками
        """
        if transactions.empty:
            return BitcoinFeatures(address=address)
        
        features = BitcoinFeatures(address=address)
        
        # Базовые признаки
        features.txs_count = len(transactions)
        first_tx_time = transactions['time'].min()
        last_tx_time = transactions['time'].max()
        features.time_span_days = (last_tx_time - first_tx_time) // 86400
        
        # Структурные признаки
        features.avg_inputs = transactions['inputs_count'].mean()
        features.median_inputs = transactions['inputs_count'].median()
        features.avg_outputs = transactions['outputs_count'].mean()
        features.median_outputs = transactions['outputs_count'].median()
        features.max_inputs = transactions['inputs_count'].max()
        features.max_outputs = transactions['outputs_count'].max()
        features.min_inputs = transactions['inputs_count'].min()
        features.min_outputs = transactions['outputs_count'].min()
        features.inputs_outputs_ratio = (
            features.avg_inputs / features.avg_outputs 
            if features.avg_outputs > 0 else 0
        )
        
        features.avg_tx_size = transactions['size'].mean()
        features.median_tx_size = transactions['size'].median()
        
        # UTXO паттерны - упрощенные вычисления
        features.avg_utxo_age = 86400.0  # Константа 1 день
        
        # Пылевые транзакции - векторизованные операции
        sent_amounts = transactions['amount_sent'].values
        received_amounts = transactions['amount_received'].values
        
        dust_sent = np.sum(sent_amounts < self.dust_threshold)
        dust_received = np.sum(received_amounts < self.dust_threshold)
        features.dust_ratio = (dust_sent + dust_received) / (len(transactions) * 2)
        
        # Крупные входы
        large_inputs = np.sum((transactions['inputs_count'] == 1) & (sent_amounts > 0))
        features.largest_input_ratio = large_inputs / len(transactions)
        
        # Множественные мелкие входы
        many_small_inputs = np.sum(
            (transactions['inputs_count'] > 5) & (sent_amounts < 0.01)
        )
        features.many_small_inputs_ratio = many_small_inputs / len(transactions)
        
        # Доля транзакций с сдачей (упрощенная оценка)
        change_transactions = np.sum(transactions['outputs_count'] > 1)
        features.change_output_ratio = change_transactions / len(transactions)
        
        # Временные паттерны
        features.transaction_frequency = len(transactions) / features.time_span_days if features.time_span_days > 0 else 0.0
        
        timestamps = transactions['time'].values
        features.burst_activity_ratio = self._calculate_burst_activity(timestamps)
        
        # Экономические паттерны
        features.avg_amount_sent = sent_amounts.mean()
        features.median_amount_sent = np.median(sent_amounts)
        features.avg_amount_received = received_amounts.mean()
        features.median_amount_received = np.median(received_amounts)
        features.max_single_tx = max(sent_amounts.max(), received_amounts.max())
        features.min_single_tx = min(sent_amounts.min(), received_amounts.min())
        
        # Круглые числа
        all_amounts = np.concatenate([sent_amounts, received_amounts])
        all_amounts = all_amounts[all_amounts > 0]
        round_count = sum(1 for amt in all_amounts if self._is_round_number(amt))
        features.round_number_ratio = round_count / len(all_amounts) if all_amounts.size > 0 else 0.0
        
        # Энтропия сумм
        features.value_entropy = self._calculate_entropy(all_amounts)
        
        # Bitcoin-специфичные признаки
        coinbase_count = transactions['is_coinbase'].sum()
        features.coinbase_ratio = (
            coinbase_count / len(transactions) 
            if len(transactions) > 0 else 0.0
        )
        
        # Сетевые паттерны - упрощенные
        features.unique_input_addresses = len(transactions)
        features.unique_output_addresses = len(transactions)
        features.address_reuse_ratio = (features.txs_count * 2) / max(
            features.unique_input_addresses + features.unique_output_addresses, 1
        )
        
        return features
    
    def process_all_addresses(
        self, 
        addresses_df: pd.DataFrame, 
        transactions_df: pd.DataFrame, 
        labels_df: pd.DataFrame
    ) -> List[BitcoinFeatures]:
        """
        Обрабатывает все адреса и извлекает признаки.
        
        Args:
            addresses_df: DataFrame с адресами
            transactions_df: DataFrame с транзакциями
            labels_df: DataFrame с метками
            
        Returns:
            Список объектов BitcoinFeatures
        """
        logger.info("Начинаем анализ транзакций...")
        
        # Объединяем адреса с метками
        merged_df = addresses_df.merge(labels_df, on='address', how='left')
        total_addresses = len(merged_df)
        
        all_features = []
        output_file = "data/bitcoin_address_features_optimized.csv"
        
        for idx, address_row in merged_df.iterrows():
            try:
                if idx % 100 == 0:
                    logger.info(f"Обработано {idx}/{total_addresses} адресов")
                
                # Получаем транзакции для адреса
                address_transactions = transactions_df[
                    transactions_df['address'] == address_row['address']
                ]
                
                # Проверяем, есть ли транзакции
                if address_transactions.empty:
                    # Пропускаем адреса без транзакций
                    logger.warning(f"Пропускаем адрес {address_row['address']} - нет транзакций")
                    continue
                else:
                    # Анализируем транзакции
                    features = self.analyze_address_transactions(
                        address_row['address'], 
                        address_transactions
                    )
                
                # Заполняем базовые поля
                features.label = address_row.get('label', '')
                
                all_features.append(features)
                
                # Сохраняем каждые SAVE_INTERVAL адресов
                if (idx > 0 and idx % self.SAVE_INTERVAL == 0):
                    self._save_features_incremental(all_features, output_file)
                    logger.info(f"Сохранено {len(all_features)} записей")
                    
            except Exception as e:
                logger.error(f"Error processing address {address_row['address']}: {e}")
                # Создаем пустой features для проблемного адреса
                empty_features = BitcoinFeatures(address=address_row['address'])
                empty_features.wallet_name = address_row.get('wallet_name', '')
                empty_features.wallet_id = address_row.get('wallet_id', '')
                empty_features.found = address_row.get('found', False)
                empty_features.updated_to_block = address_row.get('updated_to_block', '')
                empty_features.label = address_row.get('label', '')
                all_features.append(empty_features)
                continue
        
        # Финальное сохранение
        self._save_features_incremental(all_features, output_file)
        
        logger.info(f"Анализ завершен. Обработано {len(all_features)} адресов")
        return all_features
    
    def save_features_to_csv(
        self, 
        features_list: List[BitcoinFeatures], 
        output_file: str
    ) -> None:
        """
        Сохраняет извлеченные признаки в CSV файл.
        
        Args:
            features_list: Список объектов BitcoinFeatures
            output_file: Путь к выходному файлу
        """
        logger.info(f"Сохранение признаков в {output_file}...")
        
        # Преобразуем в список словарей
        data = []
        for features in features_list:
            data.append({
                'address': features.address,
                'txs_count': features.txs_count,
                'label': features.label,
                
                # Структурные признаки
                'avg_inputs': features.avg_inputs,
                'median_inputs': features.median_inputs,
                'avg_outputs': features.avg_outputs,
                'median_outputs': features.median_outputs,
                'max_inputs': features.max_inputs,
                'max_outputs': features.max_outputs,
                'min_inputs': features.min_inputs,
                'min_outputs': features.min_outputs,
                'inputs_outputs_ratio': features.inputs_outputs_ratio,
                'avg_tx_size': features.avg_tx_size,
                'median_tx_size': features.median_tx_size,
                
                # UTXO паттерны
                'avg_utxo_age': features.avg_utxo_age,
                'dust_ratio': features.dust_ratio,
                'change_output_ratio': features.change_output_ratio,
                'largest_input_ratio': features.largest_input_ratio,
                'many_small_inputs_ratio': features.many_small_inputs_ratio,
                
                # Временные паттерны
                'transaction_frequency': features.transaction_frequency,
                'burst_activity_ratio': features.burst_activity_ratio,
                'avg_block_interval': features.avg_block_interval,
                'time_span_days': features.time_span_days,
                
                # Экономические паттерны
                'avg_amount_sent': features.avg_amount_sent,
                'avg_amount_received': features.avg_amount_received,
                'total_sent': features.total_sent,
                'total_received': features.total_received,
                'max_single_tx': features.max_single_tx,
                'min_single_tx': features.min_single_tx,
                'round_number_ratio': features.round_number_ratio,
                'precision_patterns': features.precision_patterns,
                'value_entropy': features.value_entropy,
                
                # Комиссии
                'avg_fee_per_byte': features.avg_fee_per_byte,
                'fee_efficiency': features.fee_efficiency,
                'high_fee_ratio': features.high_fee_ratio,
                
                # Сетевые паттерны
                'unique_input_addresses': features.unique_input_addresses,
                'unique_output_addresses': features.unique_output_addresses,
                'address_reuse_ratio': features.address_reuse_ratio,
                'clustering_coefficient': features.clustering_coefficient,
                
                # Bitcoin-специфичные
                'coinbase_ratio': features.coinbase_ratio,
                'multisig_ratio': features.multisig_ratio,
                'op_return_usage': features.op_return_usage,
                
                # Поведенческие признаки
                'in_degree': features.in_degree,
                'out_degree': features.out_degree,
                'betweenness_centrality': features.betweenness_centrality,
                'mixing_indicators': features.mixing_indicators,
                'outlier_transactions': features.outlier_transactions
            })
        
        # Сохраняем в CSV
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Сохранено {len(data)} записей в {output_file}")
    
    def save_features_incremental(
        self, 
        features_list: List[BitcoinFeatures], 
        output_file: str,
        append_mode: bool = False
    ) -> None:
        """
        Сохраняет признаки с поддержкой инкрементального сохранения.
        
        Args:
            features_list: Список объектов BitcoinFeatures
            output_file: Путь к выходному файлу
            append_mode: Добавлять к существующему файлу
        """
        logger.info(f"Инкрементальное сохранение признаков в {output_file}...")
        
        # Преобразуем в DataFrame
        data = []
        for features in features_list:
            data.append({
                'address': features.address,
                'txs_count': features.txs_count,
                'label': features.label,
                
                # Структурные признаки
                'avg_inputs': features.avg_inputs,
                'median_inputs': features.median_inputs,
                'avg_outputs': features.avg_outputs,
                'median_outputs': features.median_outputs,
                'max_inputs': features.max_inputs,
                'max_outputs': features.max_outputs,
                'min_inputs': features.min_inputs,
                'min_outputs': features.min_outputs,
                'inputs_outputs_ratio': features.inputs_outputs_ratio,
                'avg_tx_size': features.avg_tx_size,
                'median_tx_size': features.median_tx_size,
                
                # UTXO паттерны
                'avg_utxo_age': features.avg_utxo_age,
                'dust_ratio': features.dust_ratio,
                'change_output_ratio': features.change_output_ratio,
                'largest_input_ratio': features.largest_input_ratio,
                'many_small_inputs_ratio': features.many_small_inputs_ratio,
                
                # Временные паттерны
                'transaction_frequency': features.transaction_frequency,
                'burst_activity_ratio': features.burst_activity_ratio,
                'avg_block_interval': features.avg_block_interval,
                'time_span_days': features.time_span_days,
                
                # Экономические паттерны
                'avg_amount_sent': features.avg_amount_sent,
                'avg_amount_received': features.avg_amount_received,
                'total_sent': features.total_sent,
                'total_received': features.total_received,
                'max_single_tx': features.max_single_tx,
                'min_single_tx': features.min_single_tx,
                'round_number_ratio': features.round_number_ratio,
                'precision_patterns': features.precision_patterns,
                'value_entropy': features.value_entropy,
                
                # Комиссии
                'avg_fee_per_byte': features.avg_fee_per_byte,
                'fee_efficiency': features.fee_efficiency,
                'high_fee_ratio': features.high_fee_ratio,
                
                # Сетевые паттерны
                'unique_input_addresses': features.unique_input_addresses,
                'unique_output_addresses': features.unique_output_addresses,
                'address_reuse_ratio': features.address_reuse_ratio,
                'clustering_coefficient': features.clustering_coefficient,
                
                # Bitcoin-специфичные
                'coinbase_ratio': features.coinbase_ratio,
                'multisig_ratio': features.multisig_ratio,
                'op_return_usage': features.op_return_usage,
                
                # Поведенческие признаки
                'in_degree': features.in_degree,
                'out_degree': features.out_degree,
                'betweenness_centrality': features.betweenness_centrality,
                'mixing_indicators': features.mixing_indicators,
                'outlier_transactions': features.outlier_transactions
            })
        
        # Сохраняем в CSV
        df = pd.DataFrame(data)
        
        if append_mode and os.path.exists(output_file):
            # Добавляем к существующему файлу
            df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            # Создаем новый файл
            df.to_csv(output_file, index=False)
        
        logger.info(f"Инкрементально сохранено {len(data)} записей в {output_file}")
    
    def _save_features_incremental(
        self, 
        features_list: List[BitcoinFeatures], 
        output_file: str
    ) -> None:
        """Сохраняет признаки инкрементально."""
        
        # Преобразуем в DataFrame
        data = []
        for features in features_list:
            data.append({
                'address': features.address,
                'txs_count': features.txs_count,
                'label': features.label,
                
                # Структурные признаки
                'avg_inputs': features.avg_inputs,
                'median_inputs': features.median_inputs,
                'avg_outputs': features.avg_outputs,
                'median_outputs': features.median_outputs,
                'max_inputs': features.max_inputs,
                'max_outputs': features.max_outputs,
                'min_inputs': features.min_inputs,
                'min_outputs': features.min_outputs,
                'inputs_outputs_ratio': features.inputs_outputs_ratio,
                'avg_tx_size': features.avg_tx_size,
                'median_tx_size': features.median_tx_size,
                
                # UTXO паттерны
                'avg_utxo_age': features.avg_utxo_age,
                'dust_ratio': features.dust_ratio,
                'change_output_ratio': features.change_output_ratio,
                'largest_input_ratio': features.largest_input_ratio,
                'many_small_inputs_ratio': features.many_small_inputs_ratio,
                
                # Временные паттерны
                'transaction_frequency': features.transaction_frequency,
                'burst_activity_ratio': features.burst_activity_ratio,
                'time_span_days': features.time_span_days,
                
                # Экономические паттерны
                'avg_amount_sent': features.avg_amount_sent,
                'median_amount_sent': features.median_amount_sent,
                'avg_amount_received': features.avg_amount_received,
                'median_amount_received': features.median_amount_received,
                'max_single_tx': features.max_single_tx,
                'min_single_tx': features.min_single_tx,
                'round_number_ratio': features.round_number_ratio,
                'value_entropy': features.value_entropy,
                
                # Bitcoin-специфичные
                'coinbase_ratio': features.coinbase_ratio,
                
                # Сетевые паттерны
                'unique_input_addresses': features.unique_input_addresses,
                'unique_output_addresses': features.unique_output_addresses,
                'address_reuse_ratio': features.address_reuse_ratio
            })
        
        # Сохраняем в CSV
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)


def main() -> None:
    """Основная функция для запуска оптимизированного анализа."""
    
    analyzer = BitcoinTransactionAnalyzer()
    
    # Пути к файлам
    addresses_file = "data/addresses.csv"
    transactions_file = "data/transactions.csv"
    labels_file = "data/addresses-labels.csv"
    
    try:
        # Загружаем данные
        addresses_df, transactions_df, labels_df = analyzer.load_data(
            addresses_file, transactions_file, labels_file
        )
        
        # Анализируем транзакции
        features_list = analyzer.process_all_addresses(
            addresses_df, transactions_df, labels_df
        )
        
        logger.info("Анализ завершен успешно!")
        
    except KeyboardInterrupt:
        logger.info("Анализ прерван пользователем.")
        
    except Exception as e:
        logger.error(f"Ошибка при анализе: {e}")
        raise


if __name__ == "__main__":
    main()