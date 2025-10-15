#!/usr/bin/env python3
"""
Bitcoin Data Collector for WalletExplorer.com

A robust, production-ready data collection system for Bitcoin addresses and transactions.
Features: incremental collection, parallel processing, error handling, and data persistence.

Author: Senior Python Developer
License: MIT
"""

import csv
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bitcoin_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class CollectorConfig:
    """Configuration for the Bitcoin data collector."""
    base_url: str = "https://www.walletexplorer.com"
    delay: float = 1.0
    max_workers: int = 5
    batch_size: int = 20
    transaction_batch_size: int = 100
    timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 0.3
    user_agent: str = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'


@dataclass
class AddressData:
    """Data structure for Bitcoin address information."""
    address: str
    wallet_name: str = ""
    wallet_id: str = ""
    found: bool = False
    txs_count: int = 0
    updated_to_block: str = ""
    transactions: List['TransactionData'] = field(default_factory=list)


@dataclass
class TransactionData:
    """Data structure for Bitcoin transaction information."""
    address: str
    txid: str
    amount_sent: float = 0.0
    amount_received: float = 0.0
    block_height: int = 0
    block_pos: int = 0
    time: int = 0
    balance: float = 0.0
    used_as_input: bool = False
    used_as_output: bool = False
    found: bool = False
    label: str = ""
    wallet_id: str = ""
    size: int = 0
    is_coinbase: bool = False
    updated_to_block: str = ""
    inputs_count: int = 0
    outputs_count: int = 0
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)


class WalletExplorerAPI:
    """High-level API client for WalletExplorer.com with robust error handling."""
    
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create a configured requests session with retry strategy."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update({
            'User-Agent': self.config.user_agent,
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
        })
        
        return session
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a robust HTTP request with proper error handling."""
        url = urljoin(self.config.base_url, endpoint)
        
        try:
            time.sleep(self.config.delay)
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limited, increasing delay: {url}")
                time.sleep(5)
                return self._make_request(endpoint, params)
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise
    
    def get_address_info(self, address: str) -> Tuple[AddressData, List[TransactionData]]:
        """Fetch comprehensive address information and transactions."""
        endpoint = "/api/1/address"
        params = {'address': address, 'from': 0, 'count': 100}
        
        try:
            data = self._make_request(endpoint, params)
            
            address_data = AddressData(
                address=address,
                wallet_name=data.get('label', ''),
                wallet_id=data.get('wallet_id', ''),
                found=data.get('found', False),
                txs_count=data.get('txs_count', 0),
                updated_to_block=str(data.get('updated_to_block', ''))
            )
            
            transactions: List[TransactionData] = []
            if data.get('found') and data.get('txs'):
                for tx in data['txs']:
                    transaction = TransactionData(
                        address=address,
                        txid=tx.get('txid', ''),
                        amount_sent=float(tx.get('amount_sent', 0)),
                        amount_received=float(tx.get('amount_received', 0)),
                        block_height=int(tx.get('block_height', 0)),
                        block_pos=int(tx.get('block_pos', 0)),
                        time=int(tx.get('time', 0)),
                        balance=float(tx.get('balance', 0)),
                        used_as_input=bool(tx.get('used_as_input', False)),
                        used_as_output=bool(tx.get('used_as_output', False))
                    )
                    transactions.append(transaction)
            
            return address_data, transactions
            
        except Exception as e:
            logger.error(f"Failed to fetch address info for {address}: {e}")
            return AddressData(address=address), []
    
    def get_transaction_details(self, txid: str) -> TransactionData:
        """Fetch detailed transaction information."""
        endpoint = "/api/1/tx"
        params = {'txid': txid}
        
        try:
            data = self._make_request(endpoint, params)
            
            if data.get('found'):
                inputs = data.get('in', [])
                outputs = data.get('out', [])
                
                return TransactionData(
                    address="",  # Will be set later
                    txid=txid,
                    found=True,
                    label=data.get('label', ''),
                    wallet_id=data.get('wallet_id', ''),
                    block_height=int(data.get('block_height', 0)),
                    block_pos=int(data.get('block_pos', 0)),
                    time=int(data.get('time', 0)),
                    size=int(data.get('size', 0)),
                    is_coinbase=bool(data.get('is_coinbase', False)),
                    updated_to_block=str(data.get('updated_to_block', '')),
                    inputs_count=len(inputs),
                    outputs_count=len(outputs),
                    inputs=inputs,
                    outputs=outputs
                )
            else:
                return TransactionData(address="", txid=txid, found=False)
                
        except Exception as e:
            logger.error(f"Failed to fetch transaction details for {txid}: {e}")
            return TransactionData(address="", txid=txid, found=False)


class DataPersistence:
    """Handles data persistence with atomic operations and error recovery."""
    
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.addresses_file = Path("addresses.csv")
        self.transactions_file = Path("transactions.csv")
        self.raw_responses_file = Path("raw_responses.jsonl")
        
    def load_processed_addresses(self) -> Set[str]:
        """Load already processed addresses from CSV."""
        processed: Set[str] = set()
        if self.addresses_file.exists():
            try:
                with open(self.addresses_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if address := row.get('address'):
                            processed.add(address)
                logger.info(f"Loaded {len(processed)} processed addresses from existing file")
            except Exception as e:
                logger.warning(f"Error loading processed addresses: {e}")
        return processed
    
    def load_processed_transactions(self) -> Set[str]:
        """Load already processed transaction IDs from CSV."""
        processed: Set[str] = set()
        if self.transactions_file.exists():
            try:
                with open(self.transactions_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if txid := row.get('txid'):
                            processed.add(txid)
                logger.info(f"Loaded {len(processed)} processed transactions")
            except Exception as e:
                logger.warning(f"Error loading processed transactions: {e}")
        return processed
    
    def save_addresses_batch(self, addresses: List[AddressData]) -> None:
        """Save addresses data by appending to CSV file."""
        if not addresses:
            return
            
        fieldnames = [
            'address', 'wallet_name', 'wallet_id', 'found', 
            'txs_count', 'updated_to_block', 'transactions'
        ]
        
        # Check if file exists to determine if we need to write header
        file_exists = self.addresses_file.exists()
        
        try:
            with open(self.addresses_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Write header only if file doesn't exist
                if not file_exists:
                    writer.writeheader()
                
                for addr in addresses:
                    row = {
                        'address': addr.address,
                        'wallet_name': addr.wallet_name,
                        'wallet_id': addr.wallet_id,
                        'found': addr.found,
                        'txs_count': addr.txs_count,
                        'updated_to_block': addr.updated_to_block,
                        'transactions': json.dumps([
                            {
                                'txid': tx.txid,
                                'amount_sent': tx.amount_sent,
                                'amount_received': tx.amount_received,
                                'block_height': tx.block_height,
                                'block_pos': tx.block_pos,
                                'time': tx.time,
                                'balance': tx.balance,
                                'used_as_input': tx.used_as_input,
                                'used_as_output': tx.used_as_output
                            } for tx in addr.transactions
                        ], ensure_ascii=False)
                    }
                    writer.writerow(row)
            
            logger.info(f"Appended {len(addresses)} addresses to {self.addresses_file}")
            
        except Exception as e:
            logger.error(f"Error saving addresses: {e}")
            raise
    
    def save_transactions_batch(self, transactions: List[TransactionData]) -> None:
        """Save transactions data by appending to CSV file."""
        if not transactions:
            return
            
        fieldnames = [
            'address', 'txid', 'amount_sent', 'amount_received', 'block_height',
            'block_pos', 'time', 'balance', 'used_as_input', 'used_as_output',
            'found', 'label', 'wallet_id', 'size', 'is_coinbase', 'updated_to_block',
            'inputs_count', 'outputs_count', 'inputs', 'outputs'
        ]
        
        # Check if file exists to determine if we need to write header
        file_exists = self.transactions_file.exists()
        
        try:
            with open(self.transactions_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Write header only if file doesn't exist
                if not file_exists:
                    writer.writeheader()
                
                for transaction in transactions:
                    row = {
                        'address': transaction.address,
                        'txid': transaction.txid,
                        'amount_sent': transaction.amount_sent,
                        'amount_received': transaction.amount_received,
                        'block_height': transaction.block_height,
                        'block_pos': transaction.block_pos,
                        'time': transaction.time,
                        'balance': transaction.balance,
                        'used_as_input': transaction.used_as_input,
                        'used_as_output': transaction.used_as_output,
                        'found': transaction.found,
                        'label': transaction.label,
                        'wallet_id': transaction.wallet_id,
                        'size': transaction.size,
                        'is_coinbase': transaction.is_coinbase,
                        'updated_to_block': transaction.updated_to_block,
                        'inputs_count': transaction.inputs_count,
                        'outputs_count': transaction.outputs_count,
                        'inputs': json.dumps(transaction.inputs, ensure_ascii=False),
                        'outputs': json.dumps(transaction.outputs, ensure_ascii=False)
                    }
                    writer.writerow(row)
            
            logger.info(f"Appended {len(transactions)} transactions to {self.transactions_file}")
            
        except Exception as e:
            logger.error(f"Error saving transactions: {e}")
            raise
    
    def load_transactions_from_addresses(self) -> List[TransactionData]:
        """Load all transactions from addresses.csv."""
        transactions: List[TransactionData] = []
        if not self.addresses_file.exists():
            return transactions
            
        try:
            with open(self.addresses_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if transactions_json := row.get('transactions'):
                        try:
                            tx_list = json.loads(transactions_json)
                            for tx_data in tx_list:
                                transaction = TransactionData(
                                    address=row['address'],
                                    txid=tx_data.get('txid', ''),
                                    amount_sent=float(tx_data.get('amount_sent', 0)),
                                    amount_received=float(tx_data.get('amount_received', 0)),
                                    block_height=int(tx_data.get('block_height', 0)),
                                    block_pos=int(tx_data.get('block_pos', 0)),
                                    time=int(tx_data.get('time', 0)),
                                    balance=float(tx_data.get('balance', 0)),
                                    used_as_input=bool(tx_data.get('used_as_input', False)),
                                    used_as_output=bool(tx_data.get('used_as_output', False))
                                )
                                transactions.append(transaction)
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Error parsing transactions for {row.get('address', 'unknown')}: {e}")
                            
            logger.info(f"Loaded {len(transactions)} transactions from addresses file")
            
        except Exception as e:
            logger.error(f"Error loading transactions from addresses: {e}")
            
        return transactions


class BitcoinDataCollector:
    """Main collector class orchestrating the data collection process."""
    
    def __init__(self, config: Optional[CollectorConfig] = None):
        self.config = config or CollectorConfig()
        self.api = WalletExplorerAPI(self.config)
        self.persistence = DataPersistence(self.config)
        self.processed_addresses: Set[str] = set()
        self.processed_transactions: Set[str] = set()
        
    def initialize(self) -> None:
        """Initialize the collector by loading existing data."""
        self.processed_addresses = self.persistence.load_processed_addresses()
        self.processed_transactions = self.persistence.load_processed_transactions()
        
    def collect_addresses(self, addresses: List[str]) -> None:
        """Collect address information and transactions."""
        new_addresses = [addr for addr in addresses if addr not in self.processed_addresses]
        
        if not new_addresses:
            logger.info("No new addresses to process")
            return
            
        logger.info(f"Processing {len(new_addresses)} new addresses out of {len(addresses)} total")
        
        batch: List[AddressData] = []
        for i, address in enumerate(new_addresses, 1):
            try:
                logger.info(f"Processing address {i}/{len(new_addresses)}: {address}")
                address_data, transactions = self.api.get_address_info(address)
                address_data.transactions = transactions
                batch.append(address_data)
                self.processed_addresses.add(address)
                
                # Save batch periodically
                if len(batch) >= self.config.batch_size:
                    self.persistence.save_addresses_batch(batch)
                    batch = []
                    
            except Exception as e:
                logger.error(f"Error processing address {address}: {e}")
                continue
        
        # Save remaining batch
        if batch:
            self.persistence.save_addresses_batch(batch)
            
        logger.info("Address collection completed")
    
    def collect_transaction_details(self) -> None:
        """Collect detailed information for all transactions."""
        all_transactions = self.persistence.load_transactions_from_addresses()
        unique_txids = list(set(tx.txid for tx in all_transactions if tx.txid))
        new_txids = [txid for txid in unique_txids if txid not in self.processed_transactions]
        
        if not new_txids:
            logger.info("No new transactions to process")
            return
            
        logger.info(f"Processing {len(new_txids)} new transactions out of {len(unique_txids)} total")
        
        # Process transactions in batches
        transaction_batch: List[TransactionData] = []
        processed_count = 0
        
        for i, txid in enumerate(new_txids, 1):
            try:
                logger.info(f"Processing transaction {i}/{len(new_txids)}: {txid}")
                details = self.api.get_transaction_details(txid)
                
                # Update all transactions with this txid and add first one to batch
                first_tx_added = False
                for tx in all_transactions:
                    if tx.txid == txid:
                        # Merge details
                        tx.found = details.found
                        tx.label = details.label
                        tx.wallet_id = details.wallet_id
                        tx.size = details.size
                        tx.is_coinbase = details.is_coinbase
                        tx.updated_to_block = details.updated_to_block
                        tx.inputs_count = details.inputs_count
                        tx.outputs_count = details.outputs_count
                        tx.inputs = details.inputs
                        tx.outputs = details.outputs
                        
                        # Add only first transaction to batch to avoid duplicates
                        if not first_tx_added:
                            transaction_batch.append(tx)
                            first_tx_added = True
                
                self.processed_transactions.add(txid)
                processed_count += 1
                
                # Save batch every 100 transactions
                if len(transaction_batch) >= self.config.transaction_batch_size:
                    logger.info(f"Saving batch of {len(transaction_batch)} transactions")
                    self.persistence.save_transactions_batch(transaction_batch)
                    transaction_batch = []
                    
            except Exception as e:
                logger.error(f"Error processing transaction {txid}: {e}")
                continue
        
        # Save remaining transactions in batch
        if transaction_batch:
            logger.info(f"Saving final batch of {len(transaction_batch)} transactions")
            self.persistence.save_transactions_batch(transaction_batch)
                
        logger.info("Transaction details collection completed")
    
    def collect_from_csv(self, input_file: Path) -> None:
        """Main collection method from CSV input."""
        try:
            self.initialize()
            
            # Load addresses from CSV
            addresses: List[str] = []
            with open(input_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].strip():
                        addresses.append(row[0].strip())
            
            if not addresses:
                logger.warning("No addresses found in input file")
                return
            
            # Collect addresses and transactions
            self.collect_addresses(addresses)
            
            # Collect transaction details
            self.collect_transaction_details()
            
            logger.info("Data collection completed successfully")
            
        except Exception as e:
            logger.error(f"Collection failed: {e}")
            raise
    
    def collect_parallel(self, input_file: Path) -> None:
        """Parallel collection method for improved performance."""
        try:
            self.initialize()
            
            # Load addresses from CSV
            addresses: List[str] = []
            with open(input_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].strip():
                        addresses.append(row[0].strip())
            
            if not addresses:
                logger.warning("No addresses found in input file")
                return
            
            new_addresses = [addr for addr in addresses if addr not in self.processed_addresses]
            
            if not new_addresses:
                logger.info("No new addresses to process")
            else:
                logger.info(f"Processing {len(new_addresses)} addresses with {self.config.max_workers} workers")
                
                # Parallel address collection
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    future_to_address: Dict[Future[Tuple[AddressData, List[TransactionData]]], str] = {
                        executor.submit(self.api.get_address_info, addr): addr 
                        for addr in new_addresses
                    }
                    
                    batch: List[AddressData] = []
                    for future in as_completed(future_to_address):
                        address = future_to_address[future]
                        try:
                            address_data, transactions = future.result()
                            address_data.transactions = transactions
                            batch.append(address_data)
                            self.processed_addresses.add(address)
                            
                            if len(batch) >= self.config.batch_size:
                                self.persistence.save_addresses_batch(batch)
                                batch = []
                                
                        except Exception as e:
                            logger.error(f"Error processing address {address}: {e}")
                    
                    # Save remaining batch
                    if batch:
                        self.persistence.save_addresses_batch(batch)
            
            # Collect transaction details
            self.collect_transaction_details()
            
            logger.info("Parallel data collection completed successfully")
            
        except Exception as e:
            logger.error(f"Parallel collection failed: {e}")
            raise


def main():
    """Main entry point with comprehensive argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Bitcoin Data Collector for WalletExplorer.com',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s addresses.csv
  %(prog)s addresses.csv --parallel --max-workers 10 --delay 0.5
  %(prog)s addresses.csv --batch-size 50 --transaction-batch-size 200
        """
    )
    
    parser.add_argument('input_csv', type=Path, help='Path to CSV file with Bitcoin addresses')
    parser.add_argument('--delay', type=float, default=1.0, 
                       help='Delay between requests in seconds (default: 1.0)')
    parser.add_argument('--max-workers', type=int, default=5,
                       help='Maximum number of worker threads (default: 5)')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Address batch size for saving (default: 20)')
    parser.add_argument('--transaction-batch-size', type=int, default=100,
                       help='Transaction batch size for logging (default: 100)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel processing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not args.input_csv.exists():
        logger.error(f"Input file not found: {args.input_csv}")
        return 1
    
    # Create configuration
    config = CollectorConfig(
        delay=args.delay,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        transaction_batch_size=args.transaction_batch_size,
        timeout=args.timeout
    )
    
    # Create collector and run
    collector = BitcoinDataCollector(config)
    
    try:
        if args.parallel:
            logger.info(f"Starting parallel collection with {config.max_workers} workers")
            collector.collect_parallel(args.input_csv)
        else:
            logger.info("Starting sequential collection")
            collector.collect_from_csv(args.input_csv)
        
        logger.info("Data collection completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())