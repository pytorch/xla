This directory will contain a list of reference models that we have optimized
and runs well on TPU.

Contents of this directory is organized in the following way:

- Every subdirectory is a self-contained model, as a seperate pip package.

- Each subdirectory must has a README indicating: \*\* is this training or
  inference \*\* on what devices it has been tested / developed \*\*
  instructions on running.

- Every subdirectory contains it's own set of shell scripts do with all the
  flags set for the best performance that we turned, be it training or
  inference.

- Each subdirectory can specify their own dependencies, and can depend on models
  / layers defined in well-known OSS libraries, such as HuggingFace
  transformers. But should ideally not depend on each other.

- (Optional) Each model can also have a GPU "original" version that illustrates
  and attributes where this model code came from, if any. This also helps to
  show case what changes we have done to make it performant on TPU.
