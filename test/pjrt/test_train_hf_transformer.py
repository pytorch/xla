FLAGS = {}
FLAGS['batch_size'] = 4
FLAGS['num_workers'] = 4
FLAGS['learning_rate'] = 5e-5
FLAGS['num_epochs'] = 3
FLAGS['num_cores'] = 8
FLAGS['log_steps'] = 20
FLAGS['metrics_debug'] = False
FLAGS['source_lang'] = "de"
FLAGS['target_lang'] = "en"
FLAGS['metrics_debug'] = False

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

model_name = "facebook/wmt19-de-en"
SERIAL_EXEC = xmp.MpSerialExecutor()
WRAPPED_MODEL = xmp.MpModelWrapper(FSMTForConditionalGeneration.from_pretrained(model_name))
tokenizer = FSMTTokenizer.from_pretrained(model_name)

def _train_update(device, step, loss, tracker, epoch):
  test_utils.print_training_update(
      device,
      step,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      epoch)

def finetune(rank, flags):
  torch.manual_seed(1)

  def get_dataset():
    def preprocess(examples):
      inputs = [ex[FLAGS['source_lang']] for ex in examples["translation"]]
      targets = [ex[FLAGS['target_lang']] for ex in examples["translation"]]
      return tokenizer(text=inputs, text_target=targets, padding="max_length", truncation=True)

    ds = load_dataset("news_commentary", "de-en", split="train")
    ds = ds.map(preprocess, batched=True)
    ds = ds.remove_columns(["id", "translation"])
    ds = ds.train_test_split(test_size=0.2)
    ds.set_format("torch")
    return ds["train"].shuffle(seed=42).select(range(1000)), ds["test"].shuffle(seed=42).select(range(1000))

  # Using the serial executor avoids multiple processes to
  # download the same data.
  small_train_dataset, small_test_dataset = SERIAL_EXEC.run(get_dataset)

  train_sampler = torch.utils.data.distributed.DistributedSampler(
      small_train_dataset,
      num_replicas=xm.xrt_world_size(),
      rank=xm.get_ordinal(),
      shuffle=True)
  train_loader = torch.utils.data.DataLoader(
      small_train_dataset,
      batch_size=FLAGS['batch_size'],
      sampler=train_sampler,
      num_workers=FLAGS['num_workers'],
      drop_last=True)
  test_loader = torch.utils.data.DataLoader(
      small_test_dataset,
      batch_size=FLAGS['batch_size'],
      shuffle=False,
      num_workers=FLAGS['num_workers'],
      drop_last=True)

  # Scale learning rate to world size
  lr = FLAGS['learning_rate'] * xm.xrt_world_size()

  # Get optimizer, scheduler and model
  device = xm.xla_device()
  model = WRAPPED_MODEL.to(device)
  optimizer = AdamW(model.parameters(), lr=lr)
  lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=FLAGS['num_epochs'] * len(train_loader))

  def train_loop_fn(loader):
    tracker = xm.RateTracker()
    model.train()
    for step, batch in enumerate(loader):
      optimizer.zero_grad()
      batch = {k: v.to(device) for k, v in batch.items()}
      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()
      xm.optimizer_step(optimizer)
      lr_scheduler.step()
      tracker.add(FLAGS['batch_size'])
      if step % FLAGS['log_steps'] == 0:
        xm.add_step_closure(
            _train_update, args=(device, step, loss, tracker, epoch))

  def test_loop_fn(loader):
    metric = evaluate.load("sacrebleu")
    model.eval()
    for batch in loader:
      batch = {k: v.to(device) for k, v in batch.items()}
      with torch.no_grad():
        outputs = model(**batch)

      logits = outputs.logits
      predictions = torch.argmax(logits, dim=-1)

      decoded_preds = [pred.strip() for pred in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
      decoded_labels = [[label.strip()] for label in tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)]
      metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    eval_metric = metric.compute()
    xm.mark_step()
    print('[xla:{}] Bleu={:.5f} Time={}'.format(
            xm.get_ordinal(), eval_metric["score"], time.asctime()), flush=True)

  train_device_loader = pl.MpDeviceLoader(train_loader, device)
  test_device_loader = pl.MpDeviceLoader(test_loader, device)
  for epoch in range(1, FLAGS['num_epochs'] + 1):
    xm.master_print("Started training epoch {}".format(epoch))
    train_loop_fn(train_device_loader)
    xm.master_print("Finished training epoch {}".format(epoch))

    xm.master_print("Evaluate epoch {}".format(epoch))
    test_loop_fn(test_device_loader)
    if FLAGS['metrics_debug']:
      xm.master_print(met.metrics_report(), flush=True)

if __name__ == '__main__':
  xmp.spawn(finetune, args=(FLAGS,))
