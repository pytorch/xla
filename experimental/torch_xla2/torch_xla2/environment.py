import jax


class Environment:
    """This class holds a set of configurations and "globals" needed

    for executing torch program using jax.
    Things included so far:

    op registry
    PRNGKey
    Configs

    Also helper functions to manipulate those.
    """

    _prng_key: jax.random.PRNGKey


    def __init__(self, random_seed):
        self._prng_key = jax.random.PRNGKey(random_seed)

    def get_and_rotate_prng_key(self):
        self._prng_key, key = jax.random.split(self._prng_key)


