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
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cpu\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"torch_xla2\": null, \"test\": \"eval\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cpu\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"torch_xla2\": null, \"test\": \"train\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cpu\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"torch_xla2\": null, \"test\": \"eval\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cpu\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"torch_xla2\": null, \"test\": \"train\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
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
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"torch_xla2\": null, \"test\": \"eval\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"torch_xla2\": null, \"test\": \"train\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"torch_xla2\": null, \"test\": \"eval\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"torch_xla2\": null, \"test\": \"train\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
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
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"torch_xla2\": null, \"test\": \"eval\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"torch_xla2\": null, \"test\": \"train\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
    ]
    for expected in expected_in_stderr:
      self.assertIn(expected, child.stderr)

  @absltest.skipUnless(xr.device_type() in {'CUDA'}, 'Needs CUDA accelerator')
  def test_dummy_openxla_train_cuda(self):
    child = subprocess.run([
        "python",
        EXPERIMENT_RUNNER_PY,
        "--dynamo=inductor",
        "--dynamo=openxla",
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
        "Number of selected experiment configs: 4",
        "Number of selected model configs: 1",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"torch_xla2\": null, \"test\": \"train\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"torch_xla2\": null, \"test\": \"eval\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"torch_xla2\": null, \"test\": \"eval\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"torch_xla2\": null, \"test\": \"train\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
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
        "Number of selected experiment configs: 8",
        "Number of selected model configs: 1",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": null, \"torch_xla2\": null, \"test\": \"eval\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": null, \"torch_xla2\": null, \"test\": \"train\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"torch_xla2\": null, \"test\": \"eval\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"torch_xla2\": null, \"test\": \"train\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": null, \"torch_xla2\": null, \"test\": \"eval\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": null, \"torch_xla2\": null, \"test\": \"train\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"torch_xla2\": null, \"test\": \"eval\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": null, \"xla_flags\": null, \"dynamo\": \"inductor\", \"torch_xla2\": null, \"test\": \"train\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": false}",
    ]
    for expected in expected_in_stderr:
      self.assertIn(expected, child.stderr)

  @absltest.skipUnless(xr.device_type() in {'CUDA'}, 'Needs CUDA accelerator')
  def test_dummy_dry_run_cuda_with_keep_model_data_on_cuda(self):
    child = subprocess.run([
        "python",
        EXPERIMENT_RUNNER_PY,
        "--dynamo=openxla",
        "--xla=PJRT",
        "--test=eval",
        "--test=train",
        "--suite-name=dummy",
        "--accelerator=cuda",
        "--keep-model-data-on-cuda",
        "--dry-run",
    ],
                           capture_output=True,
                           text=True)
    expected_in_stderr = [
        "Number of selected experiment configs: 2",
        "Number of selected model configs: 1",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"torch_xla2\": null, \"test\": \"eval\", \"keep_model_data_on_cuda\": true, \"enable_functionalization\": false}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"torch_xla2\": null, \"test\": \"train\", \"keep_model_data_on_cuda\": true, \"enable_functionalization\": false}",
    ]
    for expected in expected_in_stderr:
      self.assertIn(expected, child.stderr)

  def test_dummy_dry_run_with_functionalization(self):
    child = subprocess.run([
        "python",
        EXPERIMENT_RUNNER_PY,
        "--dynamo=openxla",
        "--xla=PJRT",
        "--test=eval",
        "--test=train",
        "--suite-name=dummy",
        "--accelerator=cpu",
        "--enable-functionalization",
        "--dry-run",
    ],
                           capture_output=True,
                           text=True)
    expected_in_stderr = [
        "Number of selected experiment configs: 2",
        "Number of selected model configs: 1",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cpu\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"torch_xla2\": null, \"test\": \"eval\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": true}",
        "--model-config={\"model_name\": \"dummy\"} --experiment-config={\"accelerator\": \"cpu\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"torch_xla2\": null, \"test\": \"train\", \"keep_model_data_on_cuda\": false, \"enable_functionalization\": true}",
    ]
    for expected in expected_in_stderr:
      self.assertIn(expected, child.stderr)


if __name__ == '__main__':
  unittest.main()
