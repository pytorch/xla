import argparse
import sys
from time import sleep

import torch_xla.debug.profiler as xp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Performs an on-demand profiling session on provided profiler servers."
    )

    parser.add_argument(
        "--service_addr",
        dest="service_addr",
        type=str,
        required=True,
        help='comma delimited string of addresses of the profiling servers to profile. ex. "10.0.0.2:8466" or "localhost:9012".',
    )
    parser.add_argument(
        "--logdir",
        dest="logdir",
        type=str,
        required=True,
        help='the path to write profiling output to. Both the profiler client and server must have access. ex. "gs://bucket/file/path".',
    )
    parser.add_argument(
        "--duration_ms",
        dest="duration_ms",
        type=int,
        default=10000,
        help="duration in milliseconds for tracing the server.",
    )
    parser.add_argument(
        "--start_time",
        dest="start_time",
        type=float,
        default=None,
        help=(
            "the number of seconds to sleep before starting the first profiling. "
            "This could be a floating point number for subsecond precision. "
            "Defaults to None, which skips sleeping."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--interactive",
        dest="interactive",
        type=str,
        choices=[None, "once", "loop"],
        default=None,
        help=(
            "run in interactive mode.\n"
            'If set to "once", the profiler client asks for user confirmation before starting profiling.\n'
            'If set to "loop", the profiler client repeatedly runs profiling, asking for user confirmation on each run.\n'
            "Defaults to None, which disables interactive mode."
        ),
    )

    def required_length(length):
        class RequiredLength(argparse.Action):
            def __call__(self, parser, args, values, option_string=None):
                if len(values) != length:
                    msg = f"Argument {self.dest} requires {length} arguments"
                    raise argparse.ArgumentTypeError(msg)
                setattr(args, self.dest, values)

        return RequiredLength

    group.add_argument(
        "--automatic",
        dest="automatic",
        type=int,
        nargs="+",
        default=None,
        action=required_length(2),
        help=(
            "run in automatic mode.\n"
            "Requires 2 int type arguments.\n"
            "The 1st argument specifies the number of profiles to capture.\n"
            "The 2nd argument specifies the time gap (in seconds) between the profiles, "
            "i.e. the next profiling will start X seconds after the previous profiling ends.\n"
            'ex. "--automatic 100 60" captures 100 profiles every 60 seconds.\n'
            "Defaults to None, which disables automatic mode."
        ),
    )

    return parser.parse_args()


def request_user_confirmation():
    usr_input = input('Press "Enter" to start profiling / Press "q" to exit profiling:')
    usr_input = usr_input.strip().lower()
    if usr_input == "q" or usr_input == "quit":
        print("Exiting gracefully...")
        sys.exit()
    elif usr_input:
        raise ValueError(f"Unknown user input: {usr_input}")


def main():
    args = parse_args()

    def trace():
        xp.trace(
            service_addr=args.service_addr,
            logdir=args.logdir,
            duration_ms=args.duration_ms,
        )
        print(f"Saved profiling output to {args.logdir}")

    # optionally sleep for X seconds before starting the profiling
    if args.start_time:
        print(f"Profiling will start after {args.start_time} seconds...")
        sleep(args.start_time)

    # Run performance profiling
    if args.interactive == "once":
        request_user_confirmation()
        trace()
    elif args.interactive == "loop":
        while True:
            request_user_confirmation()
            trace()
    elif args.automatic:
        num_profiles, time_gap = args.automatic
        for i in range(num_profiles):
            trace()
            if i < num_profiles - 1:
                print(f"The next profiling will start after {time_gap} seconds...")
                sleep(time_gap)
    else:
        trace()


if __name__ == "__main__":
    main()
