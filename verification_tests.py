import torch
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass
import time
import pytest

@dataclass
class TestResults:
    """Store test results and metrics"""
    output_match: bool
    performance_diff: float
    memory_usage_diff: float
    numerical_precision: float

class RefactoringVerifier:
    """Verifies correctness of refactored code against original implementation"""
    
    def __init__(self, original_module, refactored_module):
        self.original = original_module
        self.refactored = refactored_module
        torch.manual_seed(42)
        
    def generate_test_inputs(self) -> Dict[str, torch.Tensor]:
        """Generate consistent test inputs for both implementations"""
        return {
            # Weight Converter Tests
            'fp8_weights': torch.randn(1024, 1024, dtype=torch.float32).to(torch.float8_e4m3fn),
            'bf16_weights': torch.randn(1024, 1024, dtype=torch.bfloat16),
            
            # Text Generator Tests
            'prompt_tokens': torch.randint(0, 1000, (32, 128)),
            'attention_mask': torch.ones(32, 128),
            
            # FP8 Operations Tests
            'matrix_a': torch.randn(512, 512, dtype=torch.float32).to(torch.float8_e4m3fn),
'matrix_b': torch.randn(512, 512, dtype=torch.float32).to(torch.float8_e4m3fn),
            'scale_a': torch.randn(512, dtype=torch.float32),
            'scale_b': torch.randn(512, dtype=torch.float32)
        }

    def compare_outputs(self, original_output: Any, refactored_output: Any) -> bool:
        """Compare outputs from both implementations"""
        if isinstance(original_output, torch.Tensor):
            return torch.allclose(original_output, refactored_output, rtol=1e-5, atol=1e-5)
        elif isinstance(original_output, tuple):
            return all(self.compare_outputs(o, r) for o, r in zip(original_output, refactored_output))
        elif isinstance(original_output, list):
            return all(self.compare_outputs(o, r) for o, r in zip(original_output, refactored_output))
        return original_output == refactored_output

    def measure_performance(self, func, inputs: Dict[str, Any], num_runs: int = 100) -> float:
        """Measure execution time of a function"""
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = func(**inputs)
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        return (end_time - start_time) / num_runs

    def measure_memory_usage(self, func, inputs: Dict[str, Any]) -> int:
        """Measure peak memory usage of a function"""
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = func(**inputs)
        return torch.cuda.max_memory_allocated()

    def verify_weight_converter(self) -> TestResults:
        """Verify weight converter refactoring"""
        inputs = self.generate_test_inputs()
        
        # Test outputs
        original_output = self.original.act_quant(inputs['fp8_weights'])
        refactored_output = self.refactored.TensorOps.act_quant(inputs['fp8_weights'])
        output_match = self.compare_outputs(original_output, refactored_output)
        
        # Test performance
        original_perf = self.measure_performance(self.original.main, inputs)
        refactored_perf = self.measure_performance(self.refactored.main, inputs)
        perf_diff = (refactored_perf - original_perf) / original_perf * 100
        
        # Test memory usage
        original_mem = self.measure_memory_usage(self.original.main, inputs)
        refactored_mem = self.measure_memory_usage(self.refactored.main, inputs)
        mem_diff = (refactored_mem - original_mem) / original_mem * 100
        
        # Test numerical precision
        if isinstance(original_output, torch.Tensor):
            precision = torch.max(torch.abs(original_output - refactored_output))
        else:
            precision = 0.0
            
        return TestResults(output_match, perf_diff, mem_diff, precision)

    def verify_text_generator(self) -> TestResults:
        """Verify text generator refactoring"""
        inputs = self.generate_test_inputs()
        
        # Test both interactive and batch modes
        test_modes = ['interactive', 'batch']
        results = []
        
        for mode in test_modes:
            inputs['mode'] = mode
            original_output = self.original.generate(**inputs)
            refactored_output = self.refactored.generate(**inputs)
            results.append(self.compare_outputs(original_output, refactored_output))
        
        output_match = all(results)
        
        # Performance and memory tests
        perf_diff = 0.0
        mem_diff = 0.0
        precision = 0.0
        
        for mode in test_modes:
            inputs['mode'] = mode
            original_perf = self.measure_performance(self.original.generate, inputs)
            refactored_perf = self.measure_performance(self.refactored.generate, inputs)
            perf_diff += (refactored_perf - original_perf) / original_perf * 100
            
            original_mem = self.measure_memory_usage(self.original.generate, inputs)
            refactored_mem = self.measure_memory_usage(self.refactored.generate, inputs)
            mem_diff += (refactored_mem - original_mem) / original_mem * 100
            
        return TestResults(output_match, perf_diff/2, mem_diff/2, precision)

    def verify_fp8_operations(self) -> TestResults:
        """Verify FP8 operations refactoring"""
        inputs = self.generate_test_inputs()
        
        # Test matrix multiplication
        original_output = self.original.fp8_gemm(
            inputs['matrix_a'], inputs['scale_a'],
            inputs['matrix_b'], inputs['scale_b']
        )
        refactored_output = self.refactored.fp8_gemm(
            inputs['matrix_a'], inputs['scale_a'],
            inputs['matrix_b'], inputs['scale_b']
        )
        output_match = self.compare_outputs(original_output, refactored_output)
        
        # Performance test
        original_perf = self.measure_performance(self.original.fp8_gemm, inputs)
        refactored_perf = self.measure_performance(self.refactored.fp8_gemm, inputs)
        perf_diff = (refactored_perf - original_perf) / original_perf * 100
        
        # Memory test
        original_mem = self.measure_memory_usage(self.original.fp8_gemm, inputs)
        refactored_mem = self.measure_memory_usage(self.refactored.fp8_gemm, inputs)
        mem_diff = (refactored_mem - original_mem) / original_mem * 100
        
        # Numerical precision
        precision = torch.max(torch.abs(original_output - refactored_output))
        
        return TestResults(output_match, perf_diff, mem_diff, precision)

def run_verification_tests():
    """Run all verification tests and report results"""
    import kernel as owc
    import refactored_kernel as rwc
    import generate as otg
    import refactored_generate as rtg
    import fp8_cast_bf16 as ofo
    import refactored_fp8_cast_bf16 as rfo
    
    # Initialize verifiers
    weight_verifier = RefactoringVerifier(owc, rwc)
    text_verifier = RefactoringVerifier(otg, rtg)
    fp8_verifier = RefactoringVerifier(ofo, rfo)
    
    # Run tests
    weight_results = weight_verifier.verify_weight_converter()
    text_results = text_verifier.verify_text_generator()
    fp8_results = fp8_verifier.verify_fp8_operations()
    
    # Print results
    print("\n=== Weight Converter Verification ===")
    print(f"Output Match: {'✅' if weight_results.output_match else '❌'}")
    print(f"Performance Impact: {weight_results.performance_diff:+.2f}%")
    print(f"Memory Impact: {weight_results.memory_usage_diff:+.2f}%")
    print(f"Numerical Precision: {weight_results.numerical_precision:.2e}")
    
    print("\n=== Text Generator Verification ===")
    print(f"Output Match: {'✅' if text_results.output_match else '❌'}")
    print(f"Performance Impact: {text_results.performance_diff:+.2f}%")
    print(f"Memory Impact: {text_results.memory_usage_diff:+.2f}%")
    
    print("\n=== FP8 Operations Verification ===")
    print(f"Output Match: {'✅' if fp8_results.output_match else '❌'}")
    print(f"Performance Impact: {fp8_results.performance_diff:+.2f}%")
    print(f"Memory Impact: {fp8_results.memory_usage_diff:+.2f}%")
    print(f"Numerical Precision: {fp8_results.numerical_precision:.2e}")

if __name__ == "__main__":
    run_verification_tests()