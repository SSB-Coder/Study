# Deep Dive: W13 — Engine Thermodynamic Cycle Simulation

## 1. The Beginner's Analogy: The Four Strokes of Fire and Power
Imagine a closed syringe containing a breath of cool morning air:
1. **Intake & Squeeze (Stroke 1-2):** You push the plunger inward, compressing the trapped air until it is red-hot under pressure.
2. **The Spark (State 2-3):** A burst of fuel ignites, creating an explosion of blazing heat before the plunger has even begun to move.
3. **The Power Push (Stroke 3-4):** That intense pressure violently shoves the plunger backward, delivering the mechanical power that drives the vehicle forward.
4. **The Exhaust (Stroke 4-1):** The exhaust valve pops open, releasing the leftover heat into the atmosphere to start anew.
**Air-standard thermodynamic cycle simulation** tracks the exact pressure, volume, and temperature across every fraction of a millisecond.

---

## 2. The Deep Architecture/Algorithm Breakdown
Thermodynamic cycle analysis predicts internal combustion engine performance, fuel efficiency, and emissions:
1. **Air-Standard Assumptions:** The working fluid is modeled as ideal air with constant specific heats ($c_p, c_v, \gamma = 1.4$).
2. **The Otto Cycle:** Models spark-ignition gasoline engines. Heat addition is approximated as instantaneous at Constant Volume ($V_2 = V_3 = V_{\text{clearance}}$).
3. **The Diesel Cycle:** Models compression-ignition diesel engines. Fuel injection occurs over a finite crank angle, modeled as Constant Pressure ($P_2 = P_3$).
4. **Key Figures of Merit:**
   - *Net Work Output:* $W_{\text{net}} = \oint P dV = Q_{\text{in}} - Q_{\text{out}}$.
   - *Thermal Efficiency:* $\eta_{\text{th}} = W_{\text{net}} / Q_{\text{in}}$.
   - *Mean Effective Pressure (MEP):* Specific work normalized by cylinder displacement: $\text{MEP} = W_{\text{net}} / V_d$.

---

## 3. The Hardcore Mathematics

### 3.1 Isentropic Compression & Expansion
For an adiabatic, reversible ideal gas process with $\gamma = c_p / c_v$:
$$P V^\gamma = \text{constant} \implies T_2 = T_1 \left( \frac{V_1}{V_2} \right)^{\gamma - 1} = T_1 r^{\gamma - 1}$$
$$P_2 = P_1 \left( \frac{V_1}{V_2} \right)^\gamma = P_1 r^\gamma$$

### 3.2 Ideal Otto Cycle Efficiency Derivation
Heat added at constant volume:
$$Q_{\text{in}} = m c_v (T_3 - T_2)$$
Heat rejected at constant volume:
$$Q_{\text{out}} = m c_v (T_4 - T_1)$$
Thermal efficiency:
$$\eta_{\text{Otto}} = 1 - \frac{Q_{\text{out}}}{Q_{\text{in}}} = 1 - \frac{T_4 - T_1}{T_3 - T_2} = 1 - \frac{1}{r^{\gamma - 1}}$$
Efficiency depends **strictly on compression ratio $r$**!

---

## 4. The Algorithm Step-by-Step
1. **Define Geometry:** Displacement $V_d$, compression ratio $r$, intake $(P_1, T_1)$.
2. **Compute Volumes:** Clearance volume $V_c = V_d / (r - 1)$, maximum volume $V_1 = V_d + V_c$.
3. **Solve State 2:** Compute $(P_2, T_2)$ via isentropic relations.
4. **Solve State 3:** Add heat $Q_{\text{in}}$ to obtain peak combustion pressure $P_3$ and temperature $T_3$.
5. **Solve State 4:** Expand isentropically to $V_1$ to find exhaust $(P_4, T_4)$.
6. **Compute Performance:** Integrate work $W = \oint P dV$, compute MEP and thermal efficiency.

---

## 5. Limitations / When It Breaks
- **Constant Specific Heat Limitation:** Real combustion gases reach $>2500$ K where specific heat ratio drops from $\gamma=1.4$ to $\sim 1.25$ due to vibrational molecular excitation and chemical dissociation ($CO_2 \leftrightarrow CO + \frac{1}{2} O_2$), reducing real-world peak pressures by $20-30\%$.

---

## 6. Where This Fits
- **Cross-Reference Numerical Integration (V04):** Non-ideal cycle simulation integrates numerical $P-V$ indicator loops using Simpson's rule.
- **Surrogate Optimization (W14):** Tuning spark timing and compression ratio for maximum MEP without engine knock.
