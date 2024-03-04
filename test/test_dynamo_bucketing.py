import unittest
import math

import torch
import torch_xla as xla
import torch_xla.debug.metrics as met
import torch_xla.core.xla_model as xm

import torch_xla.core.dynamo_bucketing as db

class TestDynamoBucketingFunctions(unittest.TestCase):
    def test_power_of_two(self):
        expected_powers_of_two: list[int] = []
        for i in range(0, 10):
            power_of_two:int = int(math.pow(2, i))
            expected_powers_of_two.append(power_of_two)

        end:int = int(math.pow(2, 9))
        expected_index:int = 0
        for size in range(0, end + 1):
            detected_next_power_of_two:int = db._next_power_of_two(size)
            self.assertEqual(detected_next_power_of_two, expected_powers_of_two[expected_index])

            if size == expected_powers_of_two[expected_index]:
                expected_index += 1

    def test_parse_real_sizes(self):
        test_size = 50
        test_tensor = torch.randint(low = 0, high = 100, size=(test_size,))
        parsed_size, parsed_tensors = db._parse_real_sizes((test_size, test_tensor))

        self.assertEqual(parsed_size[0], test_size)
        self.assertEqual(len(parsed_size), 1)

        self.assertEqual(len(parsed_tensors), 1)
        self.assertTrue(torch.allclose(parsed_tensors[0], test_tensor))

    def test_pad_xla_args(self):
        test_tensor = torch.tensor([1, 2, 3, 4, 5])
        real_size = test_tensor.size()[0]

        padded_tensors = db.maybe_pad_xla_args((real_size, test_tensor))
        self.assertEqual(len(padded_tensors), 1)

        # Pads up to the next power of two with zeros
        expected_tensor = torch.tensor([1, 2, 3, 4, 5, 0, 0, 0])
        self.assertTrue(torch.allclose(padded_tensors[0], expected_tensor))

def add_tensor(tensor):
    return tensor.sum()

class TestDynamoXlaBridgeBucketing(unittest.TestCase):
    def test_automatic_bucketing(self):
        expected_compilation_count:int = 1
        iterations:int = 10
        max_size: int = 100
        xla_device = xm.xla_device()

        met.clear_all()
        torch._dynamo.reset()
        compiled_fn = torch.compile(add_tensor, dynamic=True, backend = "openxla")
        tensor_sum = 0

        for i in range(0, iterations):
            test_tensor = torch.randint(low = 0, high = 5, size=(max_size,))
            test_tensor = torch.nonzero(test_tensor)

            # Just for test purposes to ensure we always bucket to 128 size.
            while test_tensor.size()[0] <= 64:
                test_tensor = torch.randint(low = 0, high = 5, size=(max_size,))
                test_tensor = torch.nonzero(test_tensor)

            test_tensor = torch.flatten(test_tensor)
            tensor_sum += compiled_fn(test_tensor)

        # Expect that we only compiled once and executed 10 times via automatic bucketing.
        print(tensor_sum)
        print(met.metrics_report())
        self.assertEqual(met.metric_data('CompileTime')[0], expected_compilation_count)
        self.assertEqual(met.metric_data('ExecuteTime')[0], iterations)

if __name__ == "__main__":
    test = unittest.main()
    sys.exit(0 if test.result.wasSuccessful() else 1)
