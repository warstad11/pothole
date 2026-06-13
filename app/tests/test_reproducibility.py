
import unittest
import torch
import numpy as np
from app.core.reproducibility import set_global_seed, get_torch_generator

class TestReproducibility(unittest.TestCase):
    
    def test_numpy_reproducibility(self):
        """Test that numpy random operations are reproducible."""
        set_global_seed(42)
        arr1 = np.random.rand(10)
        
        set_global_seed(42)
        arr2 = np.random.rand(10)
        
        np.testing.assert_array_equal(arr1, arr2)
    
    def test_torch_reproducibility(self):
        """Test that torch random operations are reproducible."""
        set_global_seed(42)
        tensor1 = torch.rand(10)
        
        set_global_seed(42)
        tensor2 = torch.rand(10)
        
        torch.testing.assert_close(tensor1, tensor2)
    
    def test_generator_reproducibility(self):
        """Test that torch Generator produces same splits."""
        from torch.utils.data import TensorDataset, random_split
        
        data = TensorDataset(torch.arange(100))
        
        train1, val1 = random_split(data, [80, 20], generator=get_torch_generator(42))
        train2, val2 = random_split(data, [80, 20], generator=get_torch_generator(42))
        
        # Check same indices
        self.assertEqual(train1.indices, train2.indices)
        self.assertEqual(val1.indices, val2.indices)

if __name__ == '__main__':
    unittest.main()
