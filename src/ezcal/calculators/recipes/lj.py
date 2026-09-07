"""ASE 内蔵の Lennard-Jones。テストとデモ専用。"""

NAME = "lj"
ALIASES = ("lennardjones", "lennard-jones")
DESCRIPTION = "ASE 内蔵 Lennard-Jones (テスト用)"
REQUIRES = ()
DEFAULTS = {"epsilon": 1.0, "sigma": 1.0, "rc": 6.0}
MODELS = ()


def build(epsilon: float = 1.0, sigma: float = 1.0, rc: float = 6.0, **kwargs):
    from ase.calculators.lj import LennardJones

    return LennardJones(epsilon=epsilon, sigma=sigma, rc=rc, **kwargs)
