import torch
import torch_xla2
import jax.numpy as jnp
import jax

env = torch_xla2.default_env()
#env.config.debug_print_each_op = True
#env.config.debug_accuracy_for_each_op = True


def squeeze(): 
  with env:
    t1 = torch.tensor([-3.5])
    r1 = t1.squeeze_(-1)
    print("xla   | torch.squeeze :", r1)
  t1 = torch.tensor([-3.5])
  r1 = t1.squeeze_(-1)
  print("native| torch.squeeze :", r1)

def nanquantile(): 
  with env:
    t1 = torch.tensor([-7.0, 0.0, torch.nan])
    r1 = t1.nanquantile(0.5)
    print("xla   | torch.nanquantile(",t1,") :", r1)
  t1 = torch.tensor([-7.0, 0.0, torch.nan])
  r1 = t1.nanquantile(0.5)
  print("native| torch.nanquantile(",t1,") :", r1)

def empty(): 
  with env:
    #print("xla torch.full: ", torch.full((2,), 3))
    start1 = torch.tensor(-2)
    print("xla   | torch.tensor(-2)   ->", start1)
    start2 = torch.tensor([-2])
    print("xla   | torch.tensor([-2]) ->", start2)
    emp1 = torch.empty((1,))
    ret1 = emp1.copy_(start1)
    print("xla   | torch.empty((1,)).copy_(tensor(-2))  :", ret1)
    emp2 = torch.empty((1,))
    ret2 = emp2.copy_(start2)
    print("xla   | torch.empty((1,)).copy_(tensor([-2])):", ret2)
  
  #print("native torch.full: ", torch.full((2,), 3))
  start1 = torch.tensor(-2)
  print("native| torch.tensor(-2)   ->", start1)
  start2 = torch.tensor([-2])
  print("native| torch.tensor([-2]) ->", start2)
  emp1 = torch.empty((1,))
  ret1 = emp1.copy_(start1)
  print("native| torch.empty((1,)).copy_(tensor(-2))  :", ret1)
  emp2 = torch.empty((1,))
  ret2 = emp2.copy_(start2)
  print("native| torch.empty((1,)).copy_(tensor([-2])):", ret2)
  
def casting():
  t = torch.tensor([ 4.3000,  4.1510,  4.0020,  3.8531,  3.7041,  3.5551,  3.4061,  3.2571,
         3.1082,  2.9592,  2.8102,  2.6612,  2.5122,  2.3633,  2.2143,  2.0653,
         1.9163,  1.7673,  1.6184,  1.4694,  1.3204,  1.1714,  1.0224,  0.8735,
         0.7245,  0.5755,  0.4265,  0.2776,  0.1286, -0.0204, -0.1694, -0.3184,
        -0.4673, -0.6163, -0.7653, -0.9143, -1.0633, -1.2122, -1.3612, -1.5102,
        -1.6592, -1.8082, -1.9571, -2.1061, -2.2551, -2.4041, -2.5531, -2.7020,
        -2.8510, -3.0000])
  with env:
    print("xla   |", t.type(torch.int64))
  print("native|", t.type(torch.int64))

def linspace(): 
  dtype=torch.int64
  with env:
    print("xla   | torch.linspace(): ", torch.linspace(4.9, 3, 5, dtype=dtype))
  print("native| torch.linspace(): ", torch.linspace(4.9, 3, 5, dtype=dtype))
  return
  with env:
    print("xla   | torch.linspace(): ", torch.linspace(-2, -3, 50, dtype=dtype))
  print("native| torch.linspace(): ", torch.linspace(-2, -3, 50, dtype=dtype))
  with env:
    print("xla   | torch.linspace(): ", torch.linspace(4.3, -3, 50, dtype=dtype))
  print("native| torch.linspace(): ", torch.linspace(4.3, -3, 50, dtype=dtype))
  
def logspace(): 
  with env:
    print("xla torch.logspace: ", torch.logspace(start=-10, end=10, steps=5))
  print("native torch.logspace: ", torch.logspace(start=-10, end=10, steps=5))

def log_normal():
  with env:
    t = torch.tensor([-0.0674,  4.8280, -7.4074, -6.6235, -3.4664,  2.4134, -0.1783,  7.1360, -0.7987,  2.3815])
    print("xla  |torch.log_normal: ", t.log_normal_(0, 0.25))
  t = torch.tensor([-0.0674,  4.8280, -7.4074, -6.6235, -3.4664,  2.4134, -0.1783,  7.1360, -0.7987,  2.3815])
  print("native |torch.log_normal: ", t.log_normal_(0, 0.25))
  
def linalg_vector_norm():
  with env:
    t = torch.tensor(-0.06738138198852539)
    print("xla   | linalg.vector_norm()", torch.linalg.vector_norm(t, ord=0).dtype)
  t = torch.tensor(-0.06738138198852539)
  print("native| linalg.vector_norm()", torch.linalg.vector_norm(t, ord=0).dtype)
  
def linalg_tensorsolve():
  with env:
    A = torch.tensor([[[-0.0674,  4.8280, -7.4074, -6.6235, -3.4664,  2.4134],
         [-0.1783,  7.1360, -0.7987,  2.3815, -2.7199, -1.7691],
         [-8.5981, -5.9605, -3.7100,  0.3334,  3.5580,  5.4002]],
        [[-6.1015, -3.9192,  3.2690,  7.4735, -1.8522,  6.7348],
         [-1.4507,  0.9523,  8.1493, -8.3490, -5.6658, -2.2785],
         [-3.5082,  7.7760, -5.8336, -4.1430, -6.2878, -8.4290]]])
    B = torch.tensor([[-5.2537,  7.7364,  4.0160],
        [ 4.3621,  0.4733, -4.6142]])
    print("xla   | linalg.vectorsolve()", torch.linalg.tensorsolve(A, B))
  A = torch.tensor([[[-0.0674,  4.8280, -7.4074, -6.6235, -3.4664,  2.4134],
         [-0.1783,  7.1360, -0.7987,  2.3815, -2.7199, -1.7691],
         [-8.5981, -5.9605, -3.7100,  0.3334,  3.5580,  5.4002]],
        [[-6.1015, -3.9192,  3.2690,  7.4735, -1.8522,  6.7348],
         [-1.4507,  0.9523,  8.1493, -8.3490, -5.6658, -2.2785],
         [-3.5082,  7.7760, -5.8336, -4.1430, -6.2878, -8.4290]]])
  B = torch.tensor([[-5.2537,  7.7364,  4.0160],
        [ 4.3621,  0.4733, -4.6142]])
  print("native| linalg.vectorsolve()", torch.linalg.tensorsolve(A, B))
  
def test_lu():
  A = torch.tensor([[ 0.0437,  0.6733, -0.7089, -0.4736, -0.3145],
        [ 0.2206, -0.3749,  0.8442, -0.5197,  0.2332],
        [-0.2896, -0.6009, -0.6085, -0.9129, -0.3178]])
  print("native| lu()", torch.lu(A, pivot=True, get_infos=True))
  with env:
    A = torch.tensor([[ 0.0437,  0.6733, -0.7089, -0.4736, -0.3145],
        [ 0.2206, -0.3749,  0.8442, -0.5197,  0.2332],
        [-0.2896, -0.6009, -0.6085, -0.9129, -0.3178]])
    print("xla   | lu()", torch.lu(A, pivot=True, get_infos=True))

def test_lu_solve():
  b = torch.tensor([[ 2.3815, -2.7199, -1.7691, -8.5981],
        [-5.9605, -3.7100,  0.3334,  3.5580],
        [ 5.4002, -6.1015, -3.9192,  3.2690]])
  LU = torch.tensor([[-0.7679, -0.4551,  0.3539],
        [ 0.0390,  1.2674,  0.2928],
        [-0.0856,  0.2779, -1.2844]])
  pivots = torch.tensor([2, 3, 3], dtype=torch.int32)
  print("native| lu_solve()", torch.lu_solve(b, LU, pivots))
  with env:
    b = torch.tensor([[ 2.3815, -2.7199, -1.7691, -8.5981],
        [-5.9605, -3.7100,  0.3334,  3.5580],
        [ 5.4002, -6.1015, -3.9192,  3.2690]])
    LU = torch.tensor([[-0.7679, -0.4551,  0.3539],
        [ 0.0390,  1.2674,  0.2928],
        [-0.0856,  0.2779, -1.2844]])
    pivots = torch.tensor([2, 3, 3], dtype=torch.int32)
    print("xla   | lu_solve()", torch.lu_solve(b, LU, pivots))

def test_lu_unpack():
  unpack_data=True
  unpack_pivots=True
  if False:
    lu = torch.tensor([[-2.7199, -1.7691, -8.5981, -5.9605, -3.7100],
        [ 0.0248,  4.8718, -7.1944, -6.4758, -3.3745],
        [-0.8873, -0.3588, -3.0746, -8.4111, -2.1212]])
    pivots = torch.tensor([3, 3, 3], dtype=torch.int32)
    print("native| lu_unpack()", torch.lu_unpack(lu, pivots,unpack_data=unpack_data, unpack_pivots=unpack_pivots))
    with env:
      lu = torch.tensor([[-2.7199, -1.7691, -8.5981, -5.9605, -3.7100],
        [ 0.0248,  4.8718, -7.1944, -6.4758, -3.3745],
        [-0.8873, -0.3588, -3.0746, -8.4111, -2.1212]])
      pivots = torch.tensor([3, 3, 3], dtype=torch.int32)
      print("xla   | lu_unpack()", torch.lu_unpack(lu, pivots,unpack_data=unpack_data, unpack_pivots=unpack_pivots))

  # 2d test case
  if False:
    lu = torch.tensor([[-8.3876,  7.9964,  6.8432, -8.9778,  1.6845],
        [ 0.8269, -9.9104, -2.1215, 14.8806,  6.4389],
        [ 0.1808,  0.2953, -4.7303,  0.6897, -7.5366],
        [-0.4855, -0.7570,  0.7641,  9.0972, 16.3916],
        [ 0.1354,  0.0746, -0.2784,  0.6465, -4.7616],
        [-0.9468, -0.9447,  0.7085,  0.6482,  0.6800]])
    pivots=torch.tensor([5, 3, 6, 5, 6], dtype=torch.int32)
    print("native| lu_unpack()", torch.lu_unpack(lu, pivots,unpack_data=unpack_data, unpack_pivots=unpack_pivots))
    with env:
      lu = torch.tensor([[-8.3876,  7.9964,  6.8432, -8.9778,  1.6845],
        [ 0.8269, -9.9104, -2.1215, 14.8806,  6.4389],
        [ 0.1808,  0.2953, -4.7303,  0.6897, -7.5366],
        [-0.4855, -0.7570,  0.7641,  9.0972, 16.3916],
        [ 0.1354,  0.0746, -0.2784,  0.6465, -4.7616],
        [-0.9468, -0.9447,  0.7085,  0.6482,  0.6800]])
      pivots=torch.tensor([5, 3, 6, 5, 6], dtype=torch.int32)
      print("xla   | lu_unpack()", torch.lu_unpack(lu, pivots,unpack_data=unpack_data, unpack_pivots=unpack_pivots))
  
  # 3d testcase
  if False:
    lu = torch.tensor([[[ -5.3344,  -2.2530,  -4.3840,  -3.1485,  -7.3766],
         [  0.3589,   2.7324,  -4.2898,   0.6681,   9.0900],
         [  0.1734,   0.2346,   0.9901,   2.2108,   4.8699]],

        [[  8.5252,   5.7155,   8.5447,  -0.6509,  -8.0849],
         [ -0.5005,   8.9886,   4.2181,  -4.7992, -10.9431],
         [ -0.9880,  -0.2169,   7.5312,   3.2518,  -5.4951]],

        [[ -8.6799,   5.6140,  -7.0426,  -1.9027,  -3.6493],
         [ -0.0134,  -4.0132,   3.2959,  -8.1260,  -0.6563],
         [  0.1997,   0.7197,  -9.0417,  -1.5426,  -0.2071]]])
    pivots=torch.tensor([[1, 2, 3],
        [1, 2, 3],
        [1, 3, 3]], dtype=torch.int32)
    print("native| lu_unpack()", torch.lu_unpack(lu, pivots,unpack_data=unpack_data, unpack_pivots=unpack_pivots))
    with env:
      lu = torch.tensor([[[ -5.3344,  -2.2530,  -4.3840,  -3.1485,  -7.3766],
         [  0.3589,   2.7324,  -4.2898,   0.6681,   9.0900],
         [  0.1734,   0.2346,   0.9901,   2.2108,   4.8699]],

        [[  8.5252,   5.7155,   8.5447,  -0.6509,  -8.0849],
         [ -0.5005,   8.9886,   4.2181,  -4.7992, -10.9431],
         [ -0.9880,  -0.2169,   7.5312,   3.2518,  -5.4951]],

        [[ -8.6799,   5.6140,  -7.0426,  -1.9027,  -3.6493],
         [ -0.0134,  -4.0132,   3.2959,  -8.1260,  -0.6563],
         [  0.1997,   0.7197,  -9.0417,  -1.5426,  -0.2071]]])
      pivots=torch.tensor([[1, 2, 3],
        [1, 2, 3],
        [1, 3, 3]], dtype=torch.int32)
      print("xla   | lu_unpack()", torch.lu_unpack(lu, pivots,unpack_data=unpack_data, unpack_pivots=unpack_pivots))
  
  if True:
    lu=torch.tensor([[[[-8.3365e+00,  6.3477e+00,  4.5104e+00,  5.3273e+00,  7.6188e+00],
          [ 3.1061e-01,  3.5927e+00, -3.8981e+00, -5.0099e+00, -1.0074e-01],
          [ 5.8184e-01, -2.3698e-01,  1.3498e-01, -6.9463e+00, -1.4449e+00]],

         [[-4.5683e+00,  8.2711e+00, -2.4218e+00, -2.4573e-02, -4.3605e+00],
          [-5.9957e-01,  2.7884e+00,  1.7025e+00, -6.5343e+00, -7.9066e+00],
          [-6.9878e-01,  4.9582e-01, -1.7408e+00,  8.4385e+00, -2.2229e-02]],

         [[ 8.9846e+00,  8.7900e+00, -6.7877e+00, -7.2960e+00, -6.8219e+00],
          [-5.2339e-01,  6.5948e+00, -5.6164e+00, -8.1797e+00, -2.3340e+00],
          [-4.8310e-03, -3.4143e-01, -7.8413e+00, -6.0561e+00,  8.7051e-01]]],


        [[[-8.5389e+00, -8.6280e+00,  8.8685e+00, -5.6942e+00,  1.7255e+00],
          [-8.6666e-01, -1.3562e+01,  8.1039e+00, -8.2539e+00,  1.0327e+01],
          [ 9.0967e-02,  8.1923e-02, -3.4808e+00,  6.9131e+00, -5.7282e-01]],

         [[-8.7625e+00, -5.3132e+00, -3.0681e+00,  4.5289e+00, -5.8242e+00],
          [-9.6847e-01, -7.1503e+00, -4.5874e+00,  1.1438e+01, -1.1171e+00],
          [-8.7110e-01, -8.0781e-02, -5.7734e+00, -1.1021e+00, -5.8334e+00]],

         [[ 7.4487e+00, -3.0206e+00, -8.3463e+00,  3.6894e+00,  8.7612e+00],
          [-3.4397e-01, -8.4912e+00, -1.1034e+01,  3.5244e+00,  2.3321e+00],
          [-6.1016e-01,  2.8069e-03, -1.6438e+00,  9.3991e+00,  1.2327e+01]]],


        [[[-7.5923e+00, -3.3959e+00, -6.1899e+00,  8.5227e+00, -3.8671e+00],
          [-5.1673e-01, -9.5418e+00,  5.1356e+00,  1.2930e+01,  6.1275e+00],
          [ 1.7744e-01,  7.6855e-01, -1.0981e+01, -3.0464e+00, -4.4616e-02]],

         [[ 7.8281e+00,  4.5388e+00,  1.2742e+00,  7.6577e+00,  1.2098e+00],
          [-5.2491e-01,  7.0977e+00, -3.4945e+00, -4.1312e-01, -1.5219e-01],
          [-1.1051e-01, -9.1709e-01,  4.4391e+00, -3.5027e+00,  3.1866e+00]],

         [[-6.1305e+00,  7.8908e+00, -1.4863e+00, -8.2037e+00, -5.6577e-01],
          [-9.2197e-01,  9.6132e+00,  1.4755e+00, -6.7277e+00,  2.8337e+00],
          [ 6.7921e-01,  3.2814e-01,  2.6555e+00, -1.0014e+00, -3.1078e+00]]]])
    pivots = torch.tensor([[[1, 3, 3],
         [3, 2, 3],
         [1, 3, 3]],

        [[2, 2, 3],
         [1, 2, 3],
         [1, 2, 3]],

        [[3, 2, 3],
         [3, 3, 3],
         [2, 3, 3]]], dtype=torch.int32)
    print("native| lu_unpack()", torch.lu_unpack(lu, pivots,unpack_data=unpack_data, unpack_pivots=unpack_pivots))
    with env:
      lu=torch.tensor([[[[-8.3365e+00,  6.3477e+00,  4.5104e+00,  5.3273e+00,  7.6188e+00],
          [ 3.1061e-01,  3.5927e+00, -3.8981e+00, -5.0099e+00, -1.0074e-01],
          [ 5.8184e-01, -2.3698e-01,  1.3498e-01, -6.9463e+00, -1.4449e+00]],

         [[-4.5683e+00,  8.2711e+00, -2.4218e+00, -2.4573e-02, -4.3605e+00],
          [-5.9957e-01,  2.7884e+00,  1.7025e+00, -6.5343e+00, -7.9066e+00],
          [-6.9878e-01,  4.9582e-01, -1.7408e+00,  8.4385e+00, -2.2229e-02]],

         [[ 8.9846e+00,  8.7900e+00, -6.7877e+00, -7.2960e+00, -6.8219e+00],
          [-5.2339e-01,  6.5948e+00, -5.6164e+00, -8.1797e+00, -2.3340e+00],
          [-4.8310e-03, -3.4143e-01, -7.8413e+00, -6.0561e+00,  8.7051e-01]]],


        [[[-8.5389e+00, -8.6280e+00,  8.8685e+00, -5.6942e+00,  1.7255e+00],
          [-8.6666e-01, -1.3562e+01,  8.1039e+00, -8.2539e+00,  1.0327e+01],
          [ 9.0967e-02,  8.1923e-02, -3.4808e+00,  6.9131e+00, -5.7282e-01]],

         [[-8.7625e+00, -5.3132e+00, -3.0681e+00,  4.5289e+00, -5.8242e+00],
          [-9.6847e-01, -7.1503e+00, -4.5874e+00,  1.1438e+01, -1.1171e+00],
          [-8.7110e-01, -8.0781e-02, -5.7734e+00, -1.1021e+00, -5.8334e+00]],

         [[ 7.4487e+00, -3.0206e+00, -8.3463e+00,  3.6894e+00,  8.7612e+00],
          [-3.4397e-01, -8.4912e+00, -1.1034e+01,  3.5244e+00,  2.3321e+00],
          [-6.1016e-01,  2.8069e-03, -1.6438e+00,  9.3991e+00,  1.2327e+01]]],


        [[[-7.5923e+00, -3.3959e+00, -6.1899e+00,  8.5227e+00, -3.8671e+00],
          [-5.1673e-01, -9.5418e+00,  5.1356e+00,  1.2930e+01,  6.1275e+00],
          [ 1.7744e-01,  7.6855e-01, -1.0981e+01, -3.0464e+00, -4.4616e-02]],

         [[ 7.8281e+00,  4.5388e+00,  1.2742e+00,  7.6577e+00,  1.2098e+00],
          [-5.2491e-01,  7.0977e+00, -3.4945e+00, -4.1312e-01, -1.5219e-01],
          [-1.1051e-01, -9.1709e-01,  4.4391e+00, -3.5027e+00,  3.1866e+00]],

         [[-6.1305e+00,  7.8908e+00, -1.4863e+00, -8.2037e+00, -5.6577e-01],
          [-9.2197e-01,  9.6132e+00,  1.4755e+00, -6.7277e+00,  2.8337e+00],
          [ 6.7921e-01,  3.2814e-01,  2.6555e+00, -1.0014e+00, -3.1078e+00]]]])
      pivots = torch.tensor([[[1, 3, 3],
         [3, 2, 3],
         [1, 3, 3]],

        [[2, 2, 3],
         [1, 2, 3],
         [1, 2, 3]],

        [[3, 2, 3],
         [3, 3, 3],
         [2, 3, 3]]], dtype=torch.int32)
      print("xla   | lu_unpack()", torch.lu_unpack(lu, pivots,unpack_data=unpack_data, unpack_pivots=unpack_pivots))
    


def pivot_to_permutation():
  n = 3
  with env:
    P = jnp.array([
      [
      [[1., 0., 0.], [0.,1.,0.], [0.,0.,1.]],
      [[2., 0., 0.], [0.,2.,0.], [0.,0.,2.]],
      [[3., 0., 0.], [0.,3.,0.], [0.,0.,3.]],
      ],
      [
      [[4., 0., 0.], [0.,4.,0.], [0.,0.,4.]],
      [[5., 0., 0.], [0.,5.,0.], [0.,0.,5.]],
      [[6., 0., 0.], [0.,6.,0.], [0.,0.,6.]],
      ],
      [
      [[7., 0., 0.], [0.,7.,0.], [0.,0.,7.]],
      [[8., 0., 0.], [0.,8.,0.], [0.,0.,8.]],
      [[9., 0., 0.], [0.,9.,0.], [0.,0.,9.]],
      ],
    ]
    )
    #print("debug: start permutation matrix:", P)
    pivots = jnp.array([
      [[1, 3, 3], [3, 2, 3], [1, 3, 3]],
      [[2, 2, 3], [1, 2, 3], [1, 2, 3]],
      [[3, 2, 3], [3, 3, 3], [2, 3, 3]]], dtype=jnp.int32)
    pivot_size = pivots.shape[-1]

    def _lu_unpack_2d(p, pivot):
      jax.debug.print("unpack2d: {} {}", p , pivot)
      _pivot = pivot - 1           # pivots are offset by 1 in jax
      indices = jnp.array([*range(3)], dtype=jnp.int32)
      def update_indices(i, _indices):
        jax.debug.print("fori <<: {} {} {} {}", i, _indices, _pivot, p)
        tmp = _indices[i]
        _indices = _indices.at[i].set(_indices[_pivot[i]])
        _indices = _indices.at[_pivot[i]].set(tmp)
        jax.debug.print("fori >>: {} {} {} {}", i, _indices, _pivot, p)
        return _indices
      indices = jax.lax.fori_loop(0, _pivot.size, update_indices, indices)
      #jax.debug.print("indices {}", indices)
      p = p[jnp.array(indices)]
      p = jnp.transpose(p)
      return p

    v_lu_unpack_2d = jax.vmap(_lu_unpack_2d, in_axes=((0,None)))
    ret = v_lu_unpack_2d(P, pivots)
    print("permutation after: ", ret)
    return ret

#nanquantile()
#squeeze()
#linspace()
#casting()
#log_normal()
#linalg_vector_norm()
#linalg_tensorsolve()
#test_lu()
#test_lu_solve()
#test_lu_unpack()
pivot_to_permutation()