# Deep Dive: S06 — Orbital Mechanics (Kepler's Equation & Orbit Propagation)

## 1. The Beginner's Analogy: The Clockwork Ellipse of the Planets
Imagine an elliptical racetrack with the Sun sitting at one of the two focus points:
- If the track were a perfect circle, the planet would glide at constant speed like the second hand of a watch.
- But on an oval ellipse, Kepler's Second Law says the planet whips furiously fast when skimming close past the Sun (perihelion) and lumbers sluggishly when out at the distant apex (aphelion).
- **Kepler's Equation** links uniform "fictitious" clock time ($M$) to the true geometric position of the planet ($E$). Because it contains $\sin(E)$, you cannot isolate $E$ with ordinary algebra; you must solve it with Newton-Raphson root-finding!

---

## 2. The Deep Architecture/Algorithm Breakdown
Two-body celestial mechanics propagates orbital trajectories under inverse-square gravity:
1. **The Anomalies of Orbital Geometry:**
   - **Mean Anomaly ($M$):** Proportional to elapsed time: $M(t) = n(t - t_0)$, where mean motion $n = \sqrt{\mu / a^3}$.
   - **Eccentric Anomaly ($E$):** Auxiliary angle projected onto a circumscribed circle.
   - **True Anomaly ($\nu$):** The actual geometric angle from the periapsis to the satellite as seen from the central focus.
2. **Kepler's Equation:**
   $$M = E - e \sin E$$
   Transcendental equation linking time ($M$) to geometry ($E$).

---

## 3. The Hardcore Mathematics

### 3.1 Newton-Raphson Solution of Kepler's Equation
Define $f(E) = E - e \sin E - M = 0$.
Derivative: $f'(E) = 1 - e \cos E$.
Newton iteration:
$$E_{k+1} = E_k - \frac{E_k - e \sin E_k - M}{1 - e \cos E_k}$$
Because $e < 1$, $f'(E) \ge 1 - e > 0$, guaranteeing strict monotonicity and quadratic convergence across all elliptic orbits!

### 3.2 Orbital Position & True Anomaly
$$r = a (1 - e \cos E)$$
$$\tan\left(\frac{\nu}{2}\right) = \sqrt{\frac{1 + e}{1 - e}} \tan\left(\frac{E}{2}\right)$$
Cartesian coordinates in orbital perifocal plane:
$$\mathbf{r} = \begin{bmatrix} r \cos \nu \\ r \sin \nu \end{bmatrix}$$

### 3.3 Vis-Viva Equation
$$v^2 = \mu \left( \frac{2}{r} - \frac{1}{a} \right)$$

---

## 4. The Algorithm Step-by-Step
1. **Compute Mean Motion:** $n = \sqrt{\mu / a^3}$, period $T = 2\pi / n$.
2. **Update Mean Anomaly:** $M = (M_0 + n \Delta t) \pmod{2\pi}$.
3. **Solve Kepler:** Run Newton-Raphson to solve $E - e \sin E = M$.
4. **Compute Geometry:** Evaluate true anomaly $\nu$, radius $r$, and orbital plane vectors $(\mathbf{r}, \mathbf{v})$.

---

## 5. Limitations / When It Breaks
- **Parabolic & Hyperbolic Trajectories:** For $e \ge 1$ (interstellar comets, escape trajectories), Kepler's equation becomes hyperbolic ($M = e \sinh H - H$), requiring universal variable formulation via Stumpff functions.

---

## 6. Where This Fits
- **Cross-Reference Root Finding (V05):** Directly demonstrates Newton-Raphson on a real transcendental equation.
