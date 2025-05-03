use nannou::wgpu;

// This struct is passed to the GPU as a uniform buffer
// See alignment rules for the GPU:
// https://www.w3.org/TR/WGSL/#alignment-and-size
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FaradayData {
    pub max_iter: u32,
    pub num_particles: u32,
    _padding: [u32; 2],
    /// Initial render range in x for function
    x_range: [f32; 2],
    /// Initial render range in y for function
    y_range: [f32; 2],
}

impl Default for FaradayData {
    fn default() -> Self {
        Self {
            max_iter: 100,
            num_particles: 20_000,
            _padding: [0; 2],
            x_range: [-2.0, 0.50],
            y_range: [-1.25, 1.25],
        }
    }
}

impl FaradayData {
    /// Returns the struct as a byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { wgpu::bytes::from(self) }
    }

    /// Gets the number x_range as a tuple.
    pub fn get_x_range(&self) -> (f32, f32) {
        (self.x_range[0], self.x_range[1])
    }

    /// Gets the number y_range as a tuple.
    pub fn get_y_range(&self) -> (f32, f32) {
        (self.y_range[0], self.y_range[1])
    }

    /// Updates the the x_range field of the struct.
    pub fn update_x_range(&mut self, x_range: (f32, f32)) {
        self.x_range = [x_range.0, x_range.1];
    }

    /// Updates the the y_range field of the struct.
    pub fn update_y_range(&mut self, y_range: (f32, f32)) {
        self.y_range = [y_range.0, y_range.1];
    }
}

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
