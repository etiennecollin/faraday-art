#![allow(clippy::too_many_arguments, non_snake_case)]

/// Charging (forward) current at time t:
///
/// i_t_forward = v * C * [1 - exp( - (E_ap - E_i) / (R_s * C * v) )]
///             + v * (1.0 / R_p) * [ t_charge
///                                - R_s * C * (1.0 - exp( - (E_ap - E_i) / (R_s * C * v) )) ]
pub fn i_t_forward(
    v: f64,        // scan rate (V/s)
    C: f64,        // capacitance (F)
    E_ap: f64,     // applied potential (V)
    E_i: f64,      // initial potential (V)
    R_s: f64,      // series resistance (Ω)
    R_p: f64,      // parallel resistance (Ω)
    t_charge: f64, // charging time (s)
) -> f64 {
    let exponent = -(E_ap - E_i) / (R_s * C * v);
    let exp_term = exponent.exp();
    let capacitive = v * C * (1.0 - exp_term);
    let resistive = v * (1.0 / R_p) * (t_charge - R_s * C * (1.0 - exp_term));
    capacitive + resistive
}

/// Discharging (backward) current at time t:
///
/// i_t_backward = A
///              - v * C * [ 1 - exp( - (E_f - E_ap) / (R_s * C * v) ) ]
///              + v * (1.0 / R_p) * [ t_discharge
///                                  - R_s * C * (1.0 - exp( - (E_f - E_ap) / (R_s * C * v) )) ]
pub fn i_t_backward(
    A: f64,           // baseline constant (A)
    v: f64,           // scan rate (V/s)
    C: f64,           // capacitance (F)
    E_ap: f64,        // applied potential (V)
    E_f: f64,         // final potential (V)
    R_s: f64,         // series resistance (Ω)
    R_p: f64,         // parallel resistance (Ω)
    t_discharge: f64, // discharging time (s)
) -> f64 {
    let exponent = -(E_f - E_ap) / (R_s * C * v);
    let exp_term = exponent.exp();
    let cap_term = v * C * (1.0 - exp_term);
    let res_term = v * (1.0 / R_p) * (t_discharge - R_s * C * (1.0 - exp_term));
    A - cap_term + res_term
}

/// Gaussian CV current:
///
/// let ξ = E - E⁰;
/// i = n·F·S·k⁰·Γ⁰·exp[ -α·n·F/(R·T) · ξ ]
///     / exp[ (R·T)/(α·n·F) · (k⁰/v) · exp[ -α·n·F/(R·T) · ξ ] ]
pub fn i_gaussian(
    n: f64,      // number of electrons
    F: f64,      // Faraday constant
    S: f64,      // electrode area
    k0: f64,     // standard heterogeneous rate constant
    gamma0: f64, // surface coverage Γ⁰
    alpha: f64,  // transfer coefficient
    R: f64,      // gas constant
    T: f64,      // temperature (K)
    E: f64,      // current potential (V)
    E0: f64,     // formal potential (V)
    v: f64,      // scan rate (V/s)
) -> f64 {
    // ξ = E - E⁰
    let xi = E - E0;

    // exponent for the numerator
    let num_exp = -alpha * n * F / (R * T) * xi;
    let numerator = n * F * S * k0 * gamma0 * num_exp.exp();

    // exponent inside the denominator
    let inner = (-alpha * n * F / (R * T) * xi).exp();
    let den_exp = (R * T) / (alpha * n * F) * (k0 / v) * inner;
    let denominator = den_exp.exp();

    numerator / denominator
}
