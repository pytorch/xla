# torchax: a torch frontend for JAX, and JAX - torch interoperability layer.

**torchax** is a frontend for JAX, allowing users to write JAX programs
using PyTorch syntax.
**torchax** is also a library for providing
graph-level interoperability between PyTorch and JAX; meaning
we can reuse PyTorch models in a JAX program.

## New Location:

As of 2025-10-06, **torchax** has been permantly moved to https://github.com/google/torchax.

This file only serves as a reference. Thanks.


## Citation

```
@software{torchax,
  author = {Han Qi, Chun-nien Chan, Will Cromar, Manfei Bai, Kevin Gleanson},
  title = {torchax: PyTorch on TPU and JAX interoperability},
  url = {https://github.com/pytorch/xla/tree/master/torchax}
  version = {0.0.4},
  date = {2025-02-24},
}
```

# Maintainers & Contributors:

This library is created and maintained by the PyTorch/XLA team at Google Cloud.

It benefitted from many direct and indirect
contributions outside of the team. Many of them done by
fellow Googlers using [Google's 20% project policy](https://ebsedu.org/blog/google-tapping-workplace-actualization-20-time-rule).
Others by partner teams at Google and other companies.

Here is the list of contributors by 2025-02-25.

```
Han Qi (qihqi), PyTorch/XLA
Manfei Bai (manfeibai), PyTorch/XLA
Will Cromar (will-cromar), Meta
Milad Mohammadi (miladm), PyTorch/XLA
Siyuan Liu (lsy323), PyTorch/XLA
Bhavya Bahl (bhavya01), PyTorch/XLA
Pei Zhang (zpcore), PyTorch/XLA
Yifei Teng (tengyifei), PyTorch/XLA
Chunnien Chan (chunnienc), Google, ODML
Alban Desmaison (albanD), Meta, PyTorch
Simon Teo (simonteozw), Google (20%)
David Huang (dvhg), Google (20%)
Barni Seetharaman (barney-s), Google (20%)
Anish Karthik (anishfish2), Google (20%)
Yao Gu (guyao), Google (20%)
Yenkai Wang (yenkwang), Google (20%)
Greg Shikhman (commander), Google (20%)
Matin Akhlaghinia (matinehAkhlaghinia), Google (20%)
Tracy Chen (tracych477), Google (20%)
Matthias Guenther (mrguenther), Google (20%)
WenXin Dong (wenxindongwork), Google (20%)
Kevin Gleason (GleasonK), Google, StableHLO
Nupur Baghel (nupurbaghel), Google (20%)
Gwen Mittertreiner (gmittert), Google (20%)
Zeev Melumian (zmelumian), Lightricks
Vyom Sharma (vyom1611), Google (20%)
Shitong Wang (ShitongWang), Adobe
Rémi Doreau (ayshiff), Google (20%)
Lance Wang (wang2yn84), Google, CoreML
Hossein Sarshar (hosseinsarshar), Google (20%)
Daniel Vega-Myhre (danielvegamyhre), Google (20%)
Tianqi Fan (tqfan28), Google (20%)
Jim Lin (jimlinntu), Google (20%)
Fanhai Lu (FanhaiLu1), Google Cloud
DeWitt Clinton (dewitt), Google PyTorch
Aman Gupta (aman2930), Google (20%)
```
