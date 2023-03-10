import args_parse

from datasets import load_dataset
from transformers import FSMTForConditionalGeneration, FSMTTokenizer, get_scheduler
import torch
from torch.optim import AdamW
import evaluate
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import time

# TODO: Add a smaller model for a short integration test.
SUPPORTED_MODELS = ["facebook/wmt19-de-en"]

MODEL_OPTS = {
    '--model': {
        'choices': SUPPORTED_MODELS,
        'default': 'facebook/wmt19-de-en',
    },
    '--short_data': {
        'action': 'store_true',
    }
}


def _train_update(device, step, loss, tracker, epoch, writer):
  test_utils.print_training_update(device, step, loss.item(), tracker.rate(),
                                   tracker.global_rate(), epoch, writer)


def finetune(rank, train_dataset, test_dataset, tokenizer, flags):
  print('Starting', rank)

  if flags.short_data:
    train_dataset = train_dataset.select(range(1000))
    test_dataset = test_dataset.select(range(1000))

  train_sampler = torch.utils.data.distributed.DistributedSampler(
      train_dataset,
      num_replicas=xm.rt_world_size(),
      rank=xm.get_ordinal(),
      shuffle=True)

  # Use thread safe random number generator with a consistent seed
  rng = torch.Generator().manual_seed(42)
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=flags.batch_size,
      sampler=train_sampler,
      num_workers=flags.num_workers,
      drop_last=True,
      generator=rng)
  test_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=flags.batch_size,
      shuffle=False,
      num_workers=flags.num_workers,
      drop_last=True,
      generator=rng)

  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer(flags.logdir)

  device = xm.xla_device()
  model = FSMTForConditionalGeneration.from_pretrained(flags.model).to(device)
  optimizer = AdamW(model.parameters(), lr=flags.lr)
  lr_scheduler = get_scheduler(
      name="linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=flags.num_epochs * len(train_loader))

  def train_loop_fn(loader, epoch):
    tracker = xm.RateTracker()
    model.train()
    for step, batch in enumerate(loader):
      optimizer.zero_grad()
      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()
      xm.optimizer_step(optimizer)
      lr_scheduler.step()
      tracker.add(flags.batch_size)
      if step % flags.log_steps == 0:
        xm.add_step_closure(
            _train_update, args=(device, step, loss, tracker, epoch, writer))

  def test_loop_fn(loader, epoch):
    metric = evaluate.load("sacrebleu")
    model.eval()
    for batch in loader:
      with torch.no_grad():
        outputs = model(**batch)

      logits = outputs.logits
      predictions = torch.argmax(logits, dim=-1)

      decoded_preds = [
          pred.strip() for pred in tokenizer.batch_decode(
              predictions, skip_special_tokens=True)
      ]
      decoded_labels = [[label.strip()] for label in tokenizer.batch_decode(
          batch["labels"], skip_special_tokens=True)]
      metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    eval_metric = metric.compute()
    xm.mark_step()
    print(
        '[xla:{}] Bleu={:.5f} Time={}'.format(xm.get_ordinal(),
                                              eval_metric["score"],
                                              time.asctime()),
        flush=True)
    test_utils.write_to_summary(
        writer,
        epoch,
        dict_to_write={'bleu': eval_metric["score"]},
        write_xla_metrics=True)

  train_device_loader = pl.MpDeviceLoader(train_loader, device)
  test_device_loader = pl.MpDeviceLoader(test_loader, device)
  for epoch in range(1, flags.num_epochs + 1):
    xm.master_print("Started training epoch {}".format(epoch))
    train_loop_fn(train_device_loader, epoch)
    xm.master_print("Finished training epoch {}".format(epoch))

    xm.master_print("Evaluate epoch {}".format(epoch))
    test_loop_fn(test_device_loader, epoch)
    if flags.metrics_debug:
      xm.master_print(met.metrics_report(), flush=True)


def get_dataset(tokenizer, flags):

  def preprocess(examples):
    inputs = [ex['de'] for ex in examples["translation"]]
    targets = [ex['en'] for ex in examples["translation"]]

    return tokenizer(
        text=inputs, text_target=targets, padding="max_length", truncation=True)

  ds = load_dataset(
      "news_commentary", "de-en", data_dir=flags.datadir, split="train")
  ds = ds.map(preprocess, batched=True)
  ds = ds.remove_columns(["id", "translation"])
  ds = ds.train_test_split(test_size=0.2)
  ds.set_format("torch")

  return ds["train"], ds["test"]


if __name__ == '__main__':
  flags = args_parse.parse_common_options(
      datadir=None,
      batch_size=4,
      num_epochs=3,
      log_steps=20,
      lr=.0002,
      opts=MODEL_OPTS.items())
  tokenizer = FSMTTokenizer.from_pretrained(flags.model)
  # Load dataset once and share it with each process
  train_dataset, test_dataset = get_dataset(tokenizer, flags)
  xmp.spawn(
      finetune, args=(
          train_dataset,
          test_dataset,
          tokenizer,
          flags,
      ))
