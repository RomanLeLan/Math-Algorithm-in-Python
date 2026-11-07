"""
Integration of rational fractions: the Lazard-Rioboo-Trager algorithm. 
The integration of rational functions (i.e., functions of the form P/Q, where P and Q are two polynomials in Q[X]) is a recurring 
problem in mathematics that is often not very interesting to solve on one’s own. The Rothstein-Trager theorem, later extended by 
Lazard-Rioboo-Trager, proves that any antiderivative of a rational function can be expressed as a sum of logarithms of polynomials. 
This greatly simplifies calculations and reduces computation time. 

To use this program, enter the expression in terms of x for your rational fraction in the list myRational to be integrated, then run 
the program.

Example : I want to integrate (4+3x^4)/(4-3x^4+3x^6). I write this function in the list myRational (see line ) then I run it. The 
program return : 

  int (3*x**4 + 4)/(3*x**6 - 3*x**4 + 4) dx =
      sqrt(3)*(2*atan(sqrt(3)*x/3) + 2*atan(sqrt(3)*x**3/4 + 5*sqrt(3)*x/12) + 2*atan(3*sqrt(3)*x**5/8 - sqrt(3)*x**3/8 + sqrt(3)*x/4))/6
      verification : True

I'll leave it to you to simplify the expression on your own; it shouldn't be too complicated. I'm french so some word are in french. 
<verification : True> : the numeric computations are correct.

This algorithm was implemented by Claude of Anthropic. You can use 
this algorithm with the `Integrate` function (see the `IntegrateRational` resource function) in Mathematica. However, this computation 
can be time-consuming and may require a paid subscription. This Python program allows you to use this algorithm offline anytime, anywhere.

"""

import sympy as sp
from sympy import Poly, QQ, atan, log
from sympy.polys.matrices import DomainMatrix


# --------------------------------------------------------------------------
# 1. Réduction d'Hermite (méthode d'Horowitz-Ostrogradsky)
# --------------------------------------------------------------------------

def hermite_reduction(r, D, x):
    """
    r/D avec deg r < deg D (Poly sur Q).
    Renvoie (partie_rationnelle, e, g) avec  int r/D = c/d + int e/g,  g sans carré.

    On écrit d = pgcd(D, D'), g = D/d et on résout en (c, e) :
        r = g c' - (D'/d - g') c + d e        (deg c < deg d, deg e < deg g)
    """
    Dp = D.diff(x)
    d = D.gcd(Dp)
    if d.degree() == 0:
        return sp.Integer(0), r, D
    g = D.quo(d)
    w = Dp.quo(d) - g.diff(x)
    nc, ne, n = d.degree(), g.degree(), D.degree()

    cols = []
    for j in range(nc):
        xj = Poly(x ** j, x, domain=QQ)
        cols.append(g * xj.diff(x) - w * xj)
    for j in range(ne):
        cols.append(d * Poly(x ** j, x, domain=QQ))

    M = sp.zeros(n, n)
    for c, p in enumerate(cols):
        for i, v in enumerate(p.all_coeffs()[::-1]):
            M[i, c] = v
    rhs = sp.zeros(n, 1)
    for i, v in enumerate(r.all_coeffs()[::-1]):
        rhs[i] = v

    sol = M.LUsolve(rhs)
    c = sum(sol[j] * x ** j for j in range(nc))
    e = sum(sol[nc + j] * x ** j for j in range(ne))
    return c / d.as_expr(), Poly(e, x, domain=QQ), g


# --------------------------------------------------------------------------
# 2. Sous-résultant de degré i
# --------------------------------------------------------------------------

def subresultant(A, B, i, x):
    """
    Sous-résultant S_i(x) de A (degré exact m) et B (degré formel n = m - 1),
    0 <= i <= m. A et B sont des expressions ; B peut dépendre d'un paramètre z.

    S_i = sum_{j=0}^{i} det(M_{i,j}) x^j, où M_{i,j} est formée des m+n-2i-1
    premières colonnes de la matrice de Sylvester "tronquée" et de la colonne
    des coefficients de x^j.
    """
    A, B = Poly(A, x), Poly(B, x)
    m = A.degree()
    n = m - 1
    if i == m:
        return A.as_expr()
    if i == n:
        return B.as_expr()

    ca = A.all_coeffs()
    cb = B.all_coeffs()
    cb = [0] * (n + 1 - len(cb)) + cb
    ncols = m + n - i
    rows = []
    for s in range(n - i - 1, -1, -1):                 # x^s * A
        row = [0] * ncols
        for t, c in enumerate(ca):
            row[ncols - 1 - (s + m - t)] = c
        rows.append(row)
    for s in range(m - i - 1, -1, -1):                 # x^s * B
        row = [0] * ncols
        for t, c in enumerate(cb):
            row[ncols - 1 - (s + n - t)] = c
        rows.append(row)
    M = sp.Matrix(rows)
    nr = m + n - 2 * i

    S = 0
    for j in range(i + 1):
        cols = list(range(nr - 1)) + [ncols - 1 - j]
        dM = DomainMatrix.from_Matrix(M.extract(list(range(nr)), cols))
        S += dM.domain.to_sympy(dM.det()) * x ** j          # déterminant exact (Bareiss) sur Q[z]
    return sp.expand(S)


# --------------------------------------------------------------------------
# 3. Rioboo : logarithmes complexes -> arctangentes réelles
# --------------------------------------------------------------------------

def log_to_atan(f, g):
    """
    f, g : Poly sur un corps réel, g != 0.
    Renvoie une somme d'arctangentes de polynômes égale, à une constante près, à
        I*log(f + I*g) - I*log(f - I*g)  (= 2*atan(f/g) modulo constante)
    sans jamais diviser par un polynôme (pas de pôles parasites).
    """
    if g.is_zero:
        return sp.Integer(0)
    if f.degree() < g.degree():
        f, g = -g, f
    f, g = f.to_field(), g.to_field()
    p, r = f.div(g)
    if r.is_zero:
        return 2 * atan(p.as_expr())
    s, t, h = g.gcdex(-f)              # s g - t f = h
    u = (f * s + g * t).quo(h)
    return 2 * atan(u.as_expr()) + log_to_atan(s, t)


# --------------------------------------------------------------------------
# 4. Partie logarithmique (LRT)
# --------------------------------------------------------------------------

def _racines_simples(q):
    """
    Les racines de q sont-elles raisonnables à écrire en radicaux ?
    (degré <= 2, binôme a z^d + b avec d <= 4, ou biquadratique a z^4 + b z^2 + c).
    Sinon on garde une RootSum exacte : les formules de Cardan/Ferrari
    donneraient des expressions énormes (et très lentes à simplifier).
    """
    deg = q.degree()
    if deg <= 2:
        return True
    mon = dict(q.terms())
    exposants = {m[0] for m in mon}
    if exposants <= {0, deg}:
        return deg <= 4
    return deg == 4 and exposants <= {0, 2, 4}


def _explicit_real_form(q, coeffs, x, z):
    """
    Forme réelle de  sum_{q(a)=0} a*log(S(a,x))  quand les racines de q sont explicites.
    coeffs : coefficients de S (puissances décroissantes de x), polynômes en z.
    Renvoie None si les racines ne sont pas exploitables.
    """
    deg = q.degree()
    if not _racines_simples(q):
        return None
    rts = sp.roots(q, z)
    if sum(rts.values()) != deg:
        return None

    total = sp.Integer(0)
    for r in rts:
        u, v = sp.re(r), sp.im(r)
        if u.has(sp.re, sp.im) or v.has(sp.re, sp.im):
            return None
        cf = [sp.expand(c.subs(z, r)) for c in coeffs]
        if sp.simplify(v) == 0:                               # racine réelle
            S = sum(c * x ** (len(cf) - 1 - j) for j, c in enumerate(cf))
            total += r * log(sp.expand(S))
        elif sp.N(v) > 0:                                     # une racine par paire conjuguée
            a = sum(sp.re(c) * x ** (len(cf) - 1 - j) for j, c in enumerate(cf))
            b = sum(sp.im(c) * x ** (len(cf) - 1 - j) for j, c in enumerate(cf))
            if a.has(sp.re, sp.im) or b.has(sp.re, sp.im):
                return None
            A = Poly(sp.expand(a), x, extension=True)
            B = Poly(sp.expand(b), x, extension=True)
            # a log(A+iB) + conj(a) log(A-iB) = u log(A^2+B^2) + v * [i log((A+iB)/(A-iB))]
            total += u * log((A ** 2 + B ** 2).as_expr()) + v * log_to_atan(A, B)
    return total


def log_part(e, g, x, real=True):
    """int e/g pour g sans facteur carré, deg e < deg g, pgcd(e, g) = 1 (Polys sur Q)."""
    z = sp.Symbol("z") if x != sp.Symbol("z") else sp.Symbol("_z")
    Bz = e.as_expr() - z * g.diff(x).as_expr()

    R = Poly(sp.resultant(g.as_expr(), Bz, x), z, domain=QQ)
    _, facteurs = R.sqf_list()

    total = sp.Integer(0)
    for Ri, i in facteurs:
        if Ri.degree() < 1:
            continue
        S = subresultant(g.as_expr(), Bz, i, x)               # sous-résultant de degré i
        Sp = Poly(S, x)
        for q, _ in Ri.factor_list()[1]:                      # facteurs irréductibles sur Q
            qe = q.as_expr()
            lc = sp.rem(Sp.LC(), qe, z)
            if lc == 0:
                raise RuntimeError("coefficient dominant nul modulo q (bug ?)")
            inv = sp.invert(lc, qe, z)
            coeffs = [sp.rem(sp.expand(c * inv), qe, z) for c in Sp.all_coeffs()]
            Smon = sum(c * x ** (len(coeffs) - 1 - j) for j, c in enumerate(coeffs))

            terme = _explicit_real_form(q, coeffs, x, z) if real else None
            if terme is None:
                terme = sp.RootSum(qe, sp.Lambda(z, z * log(Smon)))
            total += terme
    return total


# --------------------------------------------------------------------------
# 5. Fonction principale
# --------------------------------------------------------------------------

def integrate_rational(f, x, real=True):
    """
    Primitive de la fraction rationnelle f(x) à coefficients rationnels.

    real=True  : forme réelle (log et atan) quand les résidus sont explicites ;
                 sinon (facteurs de degré >= 3 non triviaux) RootSum exacte.
    real=False : toujours des RootSum(R_i, z -> z*log(S_i(z, x))).
    """
    f = sp.cancel(sp.together(f))
    num, den = sp.fraction(f)
    try:
        N = Poly(num, x, domain=QQ)
        D = Poly(den, x, domain=QQ)
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
        raise ValueError("f doit être une fraction rationnelle en x à coefficients rationnels")

    q, r = N.div(D)                                           # partie polynomiale
    result = q.integrate().as_expr()
    if r.is_zero:
        return result

    rat, e, g = hermite_reduction(r, D, x)                    # partie rationnelle
    result += rat
    if e.is_zero:
        return result

    h = e.gcd(g)                                              # pôles de résidu nul
    e, g = e.quo(h), g.quo(h)
    if g.degree() == 0:
        return result
    lcg = g.LC()
    e, g = e.quo_ground(lcg), g.monic()

    return result + log_part(e, g, x, real)


# --------------------------------------------------------------------------
# Vérification numérique et démonstration
# --------------------------------------------------------------------------

def verify(f, F, x, points=(sp.Rational(3, 10), sp.Rational(17, 10), sp.Rational(-9, 4))):
    """Vérifie numériquement que F' = f en quelques points."""
    F = F.replace(lambda e: isinstance(e, sp.RootSum),
                  lambda e: sum(e.fun(r) for r in e.poly.nroots(n=30)))
    dF = sp.diff(F, x)
    return all(abs(sp.N((dF - f).subs(x, p), 20)) < 1e-12 for p in points)


if __name__ == "__main__":
    x = sp.Symbol("x")
    myRational = [
        (4+3*x**4)/(4-3*x**4+3*x**6),                        # facteur cubique : RootSum
    ]
    for f in myRational:
        F = integrate_rational(f, x)
        print(f"int {f} dx =")
        print("   ", F)
        print("    verification :", verify(f, F, x))
