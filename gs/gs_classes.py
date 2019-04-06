class Taylor:
    def __init__(self, psi, lam, rax):
        self.psi = psi
        self.lam = lam
        self.rax = rax

class GS:
    def __init__(self):
        self.ds = None
    def __str__(self):
        return "Grad-Shafranov object"
