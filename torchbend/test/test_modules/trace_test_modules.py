import torch, torch.nn as nn
from typing import List


class ReshapeFoo(nn.Module):
    test_reshape_inputs = {'x': torch.zeros(3, 5, 7)}
    test_control_inputs = {'x': torch.zeros(3, 5, 7)}
    
    @staticmethod
    def tests():
        return [(ReshapeFoo, 'test_reshape'),
                (ReshapeFoo, 'test_control')]

    def test_reshape(self, x):
        x = x.reshape(*x.shape[:-1], x.shape[-1])
        return x

    def test_control(self, x):
        if (len(x.shape) > 2):
            return torch.tensor(1)
        else:
            return torch.tensor(0)


class LogicalFlowFoo(nn.Module):
    test_logical_tensor_if_inputs = [{'x': torch.zeros(3)}, {'x': torch.ones(3)}]
    test_logical_int_if_inputs = [{'x': 0}, {'x': 1010}]

    @staticmethod
    def tests():
        return [(LogicalFlowFoo, 'test_logical_tensor_if'), 
                (LogicalFlowFoo, 'test_logical_int_if')]

    def test_logical_tensor_if(self, x: torch.Tensor):
        tensor = (x == 0)
        if tensor.all():
            return torch.tensor(0)
        else:
            return torch.tensor(1)

    def test_logical_int_if(self, x: int):
        if x == 0:
            return 0
        else:
            return 1


def get_enum(n: int):
    return range(n)

class LoopFoo(nn.Module):
    test_int_loop_inputs = [{'x': torch.arange(10), 'n': 4}]
    test_int_nested_loop_inputs = [{'x': torch.arange(10), 'n': 4}]
    test_list_loop_inputs = [{'x': [0, 1, 2, 3]}]
    test_tensor_loop_inputs = [{'x': torch.arange(12).reshape(4, 3)}]
    test_shape_loop_inputs = [{'x': torch.zeros(4, 3)}]
    test_enumerate_inputs = [{'x': torch.zeros(4, 3)}]
    test_reversed_inputs = [{'x': torch.zeros(4, 3)}]

    @staticmethod
    def tests():
        return [(LoopFoo, 'test_int_loop'), 
                (LoopFoo, 'test_int_nested_loop'), 
                (LoopFoo, 'test_tensor_loop'), 
                (LoopFoo, 'test_enumerate'), 
                (LoopFoo, 'test_shape_loop'),
                (LoopFoo, 'test_list_loop'),
                (LoopFoo, 'test_reversed')]

    def get_enum(self, n: int):
        a = range(n)
        return get_enum(n)

    def test_int_loop(self, x: torch.Tensor, n: int):
        for i in range(n):
            x = x * i
        return x

    def test_int_nested_loop(self, x: torch.Tensor, n: int):
        for i in self.get_enum(n):
            x = x * i
        return x

    def test_list_loop(self, x: List[int]):
        a = 0
        for i in x:
            a += 1
        for i in reversed(x):
            a -= 1
        return (a == 0)

    def test_tensor_loop(self, x: torch.Tensor):
        res = 0
        for t in x:
            res = res + (t % 2 == 0).int().sum()
        return res
    
    def test_shape_loop(self, x: torch.Tensor):
        for i in range(x.shape[0]):
            x[i] = i
        return x

    def test_enumerate(self, x: torch.Tensor):
        for i, x_tmp in enumerate(x):
            x[i] = i
        for i, x_tmp in enumerate(x.shape):
            x = x * x_tmp * i
        return x

    def test_reversed(self, x: torch.Tensor):
        for x_tmp in reversed(x):
            x *= x_tmp.sum()
        return x

