import unittest

import subprocess

import experiment_runner

EXPERIMENT_RUNNER_PY = experiment_runner.__file__


class ExperimentRunnerTest(unittest.TestCase):

  def test_alexnet_dry_run(self):
    child = subprocess.run([
        "python", EXPERIMENT_RUNNER_PY, "--dynamo=openxla", "--dynamo=inductor",
        "--xla=PJRT", "--xla=None", "--test=eval", "--test=train",
        "--suite-name=torchbench", "--accelerator=cpu", "--filter=^alexnet$",
        "--dry-run"
    ],
                           capture_output=True,
                           text=True)
    expected_in_stderr = [
        "Number of selected experiment configs: 2",
        "Number of selected model configs: 1",
        "'--experiment-config={\"accelerator\": \"cpu\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"test\": \"eval\"}', '--model-config={\"model_name\": \"alexnet\"}'",
        "'--experiment-config={\"accelerator\": \"cpu\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla\", \"test\": \"train\"}', '--model-config={\"model_name\": \"alexnet\"}'",
    ]
    for expected in expected_in_stderr:
      self.assertTrue(expected in child.stderr)


if __name__ == '__main__':
  unittest.main()
