from absl.testing import absltest
import unittest
import expecttest
import subprocess
import experiment_runner
import torch_xla.runtime as xr

EXPERIMENT_RUNNER_PY = experiment_runner.__file__


class ExperimentRunnerTest(expecttest.TestCase):

  def test_dummy_dry_run(self):
    child = subprocess.run([
        "python",
        EXPERIMENT_RUNNER_PY,
        "--dynamo=openxla",
        "--dynamo=inductor",
        "--xla=PJRT",
        "--xla=None",
        "--test=eval",
        "--test=train",
        "--suite-name=dummy",
        "--accelerator=cpu",
        "--dry-run",
    ],
                           capture_output=True,
                           text=True)
    expected_in_stderr = [
        "Number of selected experiment configs: 4",
        "Number of selected model configs: 1",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cpu\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"test\": \"eval\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cpu\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"test\": \"train\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cpu\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"test\": \"eval\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cpu\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"test\": \"train\"}",
    ]
    for expected in expected_in_stderr:
      self.assertIn(expected, child.stderr)

  @absltest.skipUnless(xr.device_type() in {'CUDA'}, 'Needs CUDA accelerator')
  def test_dummy_dry_run_cuda(self):
    child = subprocess.run([
        "python",
        EXPERIMENT_RUNNER_PY,
        "--dynamo=openxla",
        "--dynamo=inductor",
        "--xla=PJRT",
        "--xla=None",
        "--test=eval",
        "--test=train",
        "--suite-name=dummy",
        "--accelerator=cuda",
        "--dry-run",
    ],
                           capture_output=True,
                           text=True)
    expected_in_stderr = [
        "Number of selected experiment configs: 4",
        "Number of selected model configs: 1",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"test\": \"eval\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"test\": \"train\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"test\": \"eval\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"test\": \"train\"}",
    ]
    for expected in expected_in_stderr:
      self.assertIn(expected, child.stderr)

  @absltest.skipUnless(xr.device_type() in {'CUDA'}, 'Needs CUDA accelerator')
  def test_dummy_dry_run_inductor_cuda(self):
    child = subprocess.run([
        "python",
        EXPERIMENT_RUNNER_PY,
        "--dynamo=inductor",
        "--xla=PJRT",
        "--xla=None",
        "--test=eval",
        "--test=train",
        "--suite-name=dummy",
        "--accelerator=cuda",
        "--filter=^dummy$",
        "--dry-run",
    ],
                           capture_output=True,
                           text=True)
    expected_in_stderr = [
        "Number of selected experiment configs: 2",
        "Number of selected model configs: 1",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"test\": \"eval\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"test\": \"train\"}",
    ]
    for expected in expected_in_stderr:
      self.assertIn(expected, child.stderr)

  @absltest.skipUnless(xr.device_type() in {'CUDA'}, 'Needs CUDA accelerator')
  def test_dummy_openxla_eval_train_cuda(self):
    child = subprocess.run([
        "python",
        EXPERIMENT_RUNNER_PY,
        "--dynamo=inductor",
        "--dynamo=openxla",
        "--dynamo=openxla_eval",
        "--xla=PJRT",
        "--xla=None",
        "--test=eval",
        "--test=train",
        "--suite-name=dummy",
        "--accelerator=cuda",
        "--filter=^dummy$",
        "--dry-run",
    ],
                           capture_output=True,
                           text=True)
    expected_in_stderr = [
        "Number of selected experiment configs: 5",
        "Number of selected model configs: 1",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla_eval\", \"test\": \"eval\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"test\": \"train\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"test\": \"eval\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"test\": \"eval\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"test\": \"train\"}",
    ]
    for expected in expected_in_stderr:
      self.assertIn(expected, child.stderr)

  @absltest.skipUnless(xr.device_type() in {'CUDA'}, 'Needs CUDA accelerator')
  def test_dummy_dynamo_none_cuda(self):
    child = subprocess.run([
        "python",
        EXPERIMENT_RUNNER_PY,
        "--suite-name=dummy",
        "--accelerator=cuda",
        "--xla=PJRT",
        "--xla=None",
        "--filter=^dummy$",
        "--dry-run",
    ],
                           capture_output=True,
                           text=True)
    expected_in_stderr = [
        "Number of selected experiment configs: 9",
        "Number of selected model configs: 1",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": null, \"test\": \"eval\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": null, \"test\": \"train\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla_eval\", \"test\": \"eval\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"test\": \"eval\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"test\": \"train\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": null, \"test\": \"eval\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": null, \"test\": \"train\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"test\": \"eval\"}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"test\": \"train\"}",
    ]
    for expected in expected_in_stderr:
      self.assertIn(expected, child.stderr)


if __name__ == '__main__':
  unittest.main()
