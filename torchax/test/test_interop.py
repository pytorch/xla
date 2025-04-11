import functools
import torch
import unittest
import torchax
from torchax import interop
import torchax

class M1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.x = torch.ones(10, 10)

class M(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(100, 100)
        self.b = torch.nn.Parameter(
            torch.ones(10, 10)
        )
        c = torch.ones(10, 10)
        self.register_buffer('c', c)
        self.register_buffer('c2', c, persistent=False)
        self.d = torch.ones(10, 10)
        self.m1 = M1()



class InteropTest(unittest.TestCase):

    def setUp(self):
        torchax.enable_globally()


    def test_mod_attr(self):
        m = M()
        params, buffers = interop.extract_all_buffers(m)
        self.assertEqual(
            set(params.keys()), {'a.weight', 'a.bias', 'b'}
        )
        self.assertEqual(
            set(buffers.keys()), {'c', 'c2', 'd', 'm1.x'}
        )

        interop.set_all_buffers(m, {'a.weight': torch.tensor([0.0])}, 
                                   {'m1.x': torch.tensor([0.0])})
        self.assertEqual(m.a.weight.item(), 0)
        self.assertEqual(m.m1.x.item(), 0)

    def test_j2t_autograd_forward(self):
        with torchax.default_env():
            # Setup
            def fn(x):
                return x + 1

            j2t_fn = interop.j2t_autograd(fn)
            x = torch.ones(2, 2, requires_grad=True, device='jax')
            
            # Act
            actual = j2t_fn(x)
            
            # Assert
            expected = torch.ones(2, 2) + 1
            torch.testing.assert_close(actual, expected, check_device=False)

    def test_j2t_autograd_backward(self):
        with torchax.default_env():
            # Setup
            def fn(x):
                return x * 2

            j2t_fn = interop.j2t_autograd(fn)
            x = torch.ones(2, 2, device='jax').requires_grad_()
            
            # Act
            actual = j2t_fn(x)
            actual.sum().backward()
            
            # Assert
            expected = torch.ones(2, 2) * 2
            torch.testing.assert_close(x.grad, expected, check_device=False)

    def test_module_with_shared_weights(self):

        class M2(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(10, 10)
                self.b = self.a
            
            def forward(self, x):
                return self.a(self.b(x))

        m = M2().to('jax')

        m_jitted = interop.JittableModule(m, dedup_parameters=True)


        # a's weights and bias and b's weights and bias
        self.assertEqual(len(m.state_dict()), 4)
        
        # b's weights and bias are deduped
        self.assertEqual(len(m_jitted.params), 2)

        x = torch.randn(10, 10).to('jax')

        expected = m(x)
        
        torch.testing.assert_allclose(m_jitted(x), expected)

        # make sure buffer donation works
        functional_forward = interop.jax_jit(
            functools.partial(m_jitted.functional_call, 'forward'),
            kwargs_for_jax_jit={
                'donate_argnums': (0, )
            }
        )

        torch.testing.assert_allclose(
            functional_forward(m_jitted.params, m_jitted.buffers, x) , expected)


        



if __name__ == '__main__':
    unittest.main()
