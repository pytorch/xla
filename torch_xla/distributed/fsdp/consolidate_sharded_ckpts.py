from argparse import ArgumentParser

from .state_dict_utils import consolidate_sharded_model_checkpoints


def main():
  parser = ArgumentParser()
  parser.add_argument(
      "--ckpt_prefix",
      type=str,
      required=True,
      help=(
          "The path prefix of the XLA FSDP checkpoint files to be consolidated. "
          "Files matching the pattern ``ckpt_prefix + ckpt_suffix`` will be loaded."
      ),
  )
  parser.add_argument(
      "--ckpt_suffix",
      type=str,
      default="*.pth",
      help=(
          "The path suffix of the XLA FSDP checkpoint files to be consolidated. "
          "Files matching the pattern ``ckpt_prefix + ckpt_suffix`` will be loaded."
      ),
  )
  parser.add_argument(
      "--save_path",
      type=str,
      default="",
      help=("The save path of the output consolidated model state dict "
            "(default is ``ckpt_prefix + '_consolidated.pth'``)"),
  )
  args = parser.parse_args()
  consolidate_sharded_model_checkpoints(args.ckpt_prefix, args.ckpt_suffix,
                                        args.save_path)


if __name__ == "__main__":
  main()
