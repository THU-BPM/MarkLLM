# ===================================================
# mersenne.py
# Description: Implementation of Mersenne Twister RNG
# ===================================================

class MersenneRNG(object):
    """Mersenne Twister RNG implementation."""

    def __init__(self, seed: int = 42):
        """
            Initialize the Mersenne Twister RNG.

            Parameters:
                seed (int): Seed for the RNG.
        """
        self.state = [0] * 624
        self.f = 1812433253
        self.m = 397
        self.u = 11
        self.s = 7
        self.b = 0x9D2C5680
        self.t = 15
        self.c = 0xEFC60000
        self.l = 18
        self.index = 624
        self.lower_mask = (1 << 31) - 1
        self.upper_mask = 1 << 31

        # update state
        self.state[0] = seed
        for i in range(1, 624):
            self.state[i] = self.int_32(self.f * (self.state[i - 1] ^ (self.state[i - 1] >> 30)) + i)

    def twist(self) -> None:
        """Twist the RNG."""
        for i in range(624):
            temp = self.int_32((self.state[i] & self.upper_mask) + (self.state[(i + 1) % 624] & self.lower_mask))
            temp_shift = temp >> 1
            if temp % 2 != 0:
                temp_shift = temp_shift ^ 0x9908b0df
            self.state[i] = self.state[(i + self.m) % 624] ^ temp_shift
        self.index = 0

    def int_32(self, number: int) -> int:
        """Return 32-bit integer."""
        return int(0xFFFFFFFF & number)

    def randint(self) -> int:
        """Return random integer."""

        if self.index >= 624:
            self.twist()
        y = self.state[self.index]
        y = y ^ (y >> self.u)
        y = y ^ ((y << self.s) & self.b)
        y = y ^ ((y << self.t) & self.c)
        y = y ^ (y >> self.l)
        self.index += 1
        return self.int_32(y)

    def rand(self) -> float:
        """Return random float."""
        return self.randint() * (1.0 / 4294967296.0)

    def randperm(self, n: int) -> list[int]:
        """Return random permutation of n elements."""
        
        # Fisher-Yates shuffle
        p = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.randint() % i
            p[i], p[j] = p[j], p[i]

        return p
