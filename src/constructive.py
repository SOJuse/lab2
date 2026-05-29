from fractions import Fraction


class ConstructiveNumber:
    def __init__(self, a, b):
        self.a = Fraction(a).limit_denominator(10**9)
        self.b = Fraction(b).limit_denominator(10**9)

    @classmethod
    def from_real(cls, x, eps):
        return cls(x - eps, x + eps)

    def get(self, alpha=0.5):
        return float(self.a + alpha * (self.b - self.a))

    def eps(self):
        return float((self.b - self.a) / 2)

    def __add__(self, other):
        if isinstance(other, ConstructiveNumber):
            return ConstructiveNumber(self.a + other.a, self.b + other.b)
        o = Fraction(other).limit_denominator(10**9)
        return ConstructiveNumber(self.a + o, self.b + o)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ConstructiveNumber):
            return ConstructiveNumber(self.a - other.b, self.b - other.a)
        o = Fraction(other).limit_denominator(10**9)
        return ConstructiveNumber(self.a - o, self.b - o)

    def __rsub__(self, other):
        o = Fraction(other).limit_denominator(10**9)
        return ConstructiveNumber(o - self.b, o - self.a)

    def __mul__(self, other):
        if isinstance(other, ConstructiveNumber):
            ps = [self.a * other.a, self.a * other.b,
                  self.b * other.a, self.b * other.b]
            return ConstructiveNumber(min(ps), max(ps))
        o = Fraction(other).limit_denominator(10**9)
        if o >= 0:
            return ConstructiveNumber(self.a * o, self.b * o)
        return ConstructiveNumber(self.b * o, self.a * o)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, ConstructiveNumber):
            if other.a <= 0 <= other.b:
                raise ValueError("division by interval containing zero")
            qs = [self.a / other.a, self.a / other.b,
                  self.b / other.a, self.b / other.b]
            return ConstructiveNumber(min(qs), max(qs))
        o = Fraction(other).limit_denominator(10**9)
        if o == 0:
            raise ValueError("division by zero")
        if o > 0:
            return ConstructiveNumber(self.a / o, self.b / o)
        return ConstructiveNumber(self.b / o, self.a / o)

    def __neg__(self):
        return ConstructiveNumber(-self.b, -self.a)

    def __lt__(self, other):
        if isinstance(other, ConstructiveNumber):
            return self.b < other.a
        return self.b < other

    def __gt__(self, other):
        if isinstance(other, ConstructiveNumber):
            return self.a > other.b
        return self.a > other

    def __float__(self):
        return self.get(0.5)

    def __repr__(self):
        return f"CN([{float(self.a):.6g}, {float(self.b):.6g}], eps={self.eps():.3e})"
