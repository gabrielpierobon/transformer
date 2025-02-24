import numpy as np
from typing import Tuple, List
from .config import DataConfig

class SequenceGenerator:
    """Generates sequences for model training."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
    def create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from time series data.
        
        Args:
            data: Input time series data
            sequence_length: Length of sequences to generate
            
        Returns:
            Tuple of input sequences and target values
        """
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            sequence = data[i:(i + sequence_length)]
            target = data[i + sequence_length]
            sequences.append(sequence)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)
    
    def create_padded_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create padded sequences for masked training.
        
        Args:
            data: Input time series data
            sequence_length: Maximum sequence length
            
        Returns:
            Tuple of padded sequences and targets
        """
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            # Create sub-sequences with padding
            for j in range(1, sequence_length + 1):
                padded_sequence = np.zeros(sequence_length)
                sequence = data[i:i + j]
                padded_sequence[-j:] = sequence
                target = data[i + sequence_length]
                
                sequences.append(padded_sequence)
                targets.append(target)
                
        return np.array(sequences), np.array(targets)