def test_sentencepiece_interop():
  import os
  if not os.path.exists("/tmp/test_model.model"):
    import urllib.request
    urllib.request.urlretrieve(
        "https://github.com/google/sentencepiece/raw/refs/heads/master/python/test/test_model.model",
        "/tmp/test_model.model")
  import torch_xla
  import sentencepiece as spm
  sp_model = spm.SentencePieceProcessor("/tmp/test_model.model")
