import unittest
from torchax import interop
import torch


class MyAwesomeModel(torch.nn.Module):
  pass


class EvenMoreAwesomeModel(torch.nn.Module):
  pass


class JittableModuleTest(unittest.TestCase):

  def test_isinstance_works(self):

    # Export and check for composite operations
    model = MyAwesomeModel()
    jittable_module = interop.JittableModule(model)

    # jittable_module should remain an instance of MyAwesomeModel logicailly
    assert isinstance(jittable_module, MyAwesomeModel)

  def test_isinstance_does_not_mix(self):

    # Export and check for composite operations
    JittableAwesomeModel = interop.JittableModule(MyAwesomeModel())
    JittableMoreAwesomeModel = interop.JittableModule(EvenMoreAwesomeModel())

    # jittable_module should remain an instance of MyAwesomeModel logicailly
    assert isinstance(JittableAwesomeModel, MyAwesomeModel)
    assert not isinstance(JittableAwesomeModel, EvenMoreAwesomeModel)
    assert isinstance(JittableMoreAwesomeModel, EvenMoreAwesomeModel)
    assert not isinstance(JittableMoreAwesomeModel, MyAwesomeModel)


if __name__ == '__main__':
  unittest.main()
