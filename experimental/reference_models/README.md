This directory will contain a list of reference models that 
we have optimized and runs well with torch_xla or torch_xla2.

Contents of this directory is organized in the following way:

* Every subdirectory is a self-contained model, as a seperate pip package.
* Every subdirectory contains it's own set of shell scripts do with all the flags
  set for the best performance that we turned, be it training or inference.
* Each subdirectory can specify their own dependencies, and can depend on models / layers
  defined in well-known OSS libraries, such as HuggingFace transformers.


