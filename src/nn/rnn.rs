//! Recurrent Neural Networks
use crate::{Device, Kind, Tensor, ToDevice};

/// Trait for Recurrent Neural Networks.
pub trait RNN {
    type State;

    /// A zero state from which the recurrent network is usually initialized.
    fn zero_state(&self, batch_dim: i64) -> Self::State;

    /// Applies a single step of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, features].
    fn step(&self, input: &Tensor, state: &Self::State) -> Self::State;

    /// Applies multiple steps of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, seq_len, features].
    /// The initial state is the result of applying zero_state.
    fn seq(&self, input: &Tensor) -> (Tensor, Self::State) {
        let batch_dim = input.size()[0];
        let state = self.zero_state(batch_dim);
        self.seq_init(input, &state)
    }

    /// Applies multiple steps of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, seq_len, features].
    fn seq_init(&self, input: &Tensor, state: &Self::State) -> (Tensor, Self::State);
}

/// The state for a LSTM network, this contains two tensors.
#[derive(Debug)]
pub struct LSTMState(pub (Tensor, Tensor));

impl LSTMState {
    /// The hidden state vector, which is also the output of the LSTM.
    pub fn h(&self) -> Tensor {
        (self.0).0.shallow_clone()
    }

    /// The cell state vector.
    pub fn c(&self) -> Tensor {
        (self.0).1.shallow_clone()
    }
}

// The GRU and LSTM layers share the same config.
/// Configuration for the GRU and LSTM layers.
#[derive(Debug, Clone, Copy)]
pub struct RNNConfig {
    pub has_biases: bool,
    pub num_layers: i64,
    pub dropout: f64,
    pub train: bool,
    pub bidirectional: bool,
    pub batch_first: bool,
}

impl Default for RNNConfig {
    fn default() -> Self {
        RNNConfig {
            has_biases: true,
            num_layers: 1,
            dropout: 0.,
            train: true,
            bidirectional: false,
            batch_first: true,
        }
    }
}

/// A Long Short-Term Memory (LSTM) layer.
///
/// https://en.wikipedia.org/wiki/Long_short-term_memory
#[derive(Debug)]
pub struct LSTM {
    flat_weights: Vec<Tensor>,
    hidden_dim: i64,
    config: RNNConfig,
    device: Device,
}

/// Creates a LSTM layer.
pub fn lstm(vs: &super::var_store::Path, in_dim: i64, hidden_dim: i64, c: RNNConfig) -> LSTM {
    let num_directions = if c.bidirectional { 2 } else { 1 };
    let gate_dim = 4 * hidden_dim;
    let mut flat_weights = vec![];
    for layer_idx in 0..c.num_layers {
        for _direction_idx in 0..num_directions {
            let in_dim = if layer_idx == 0 {
                in_dim
            } else {
                hidden_dim * num_directions
            };
            let w_ih = vs.kaiming_uniform("w_ih", &[gate_dim, in_dim]);
            let w_hh = vs.kaiming_uniform("w_hh", &[gate_dim, hidden_dim]);
            let b_ih = vs.zeros("b_ih", &[gate_dim]);
            let b_hh = vs.zeros("b_hh", &[gate_dim]);
            flat_weights.push(w_ih);
            flat_weights.push(w_hh);
            flat_weights.push(b_ih);
            flat_weights.push(b_hh);
        }
    }
    LSTM {
        flat_weights,
        hidden_dim,
        config: c,
        device: vs.device(),
    }
}

impl RNN for LSTM {
    type State = LSTMState;

    fn zero_state(&self, batch_dim: i64) -> LSTMState {
        let num_directions = if self.config.bidirectional { 2 } else { 1 };
        let layer_dim = self.config.num_layers * num_directions;
        let shape = [layer_dim, batch_dim, self.hidden_dim];
        let zeros = Tensor::zeros(&shape, (Kind::Float, self.device));
        LSTMState((zeros.shallow_clone(), zeros.shallow_clone()))
    }

    fn step(&self, input: &Tensor, in_state: &LSTMState) -> LSTMState {
        let input = input.unsqueeze(1);
        let (_output, state) = self.seq_init(&input, in_state);
        state
    }

    fn seq_init(&self, input: &Tensor, in_state: &LSTMState) -> (Tensor, LSTMState) {
        let LSTMState((h, c)) = in_state;
        let flat_weights = self.flat_weights.iter().map(|x| x).collect::<Vec<_>>();
        let (output, h, c) = input.lstm(
            &[h, c],
            &flat_weights,
            self.config.has_biases,
            self.config.num_layers,
            self.config.dropout,
            self.config.train,
            self.config.bidirectional,
            self.config.batch_first,
        );
        (output, LSTMState((h, c)))
    }
}

impl ToDevice for LSTM {
    fn to_device(&self, device: Device) -> Self {
        Self {
            flat_weights: self
                .flat_weights
                .iter()
                .map(|x| x.to_device(device))
                .collect::<Vec<_>>(),
            hidden_dim: self.hidden_dim,
            config: self.config,
            device,
        }
    }
}

/// A GRU state, this contains a single tensor.
#[derive(Debug)]
pub struct GRUState(pub Tensor);

impl GRUState {
    pub fn value(&self) -> Tensor {
        self.0.shallow_clone()
    }
}

/// A Gated Recurrent Unit (GRU) layer.
///
/// https://en.wikipedia.org/wiki/Gated_recurrent_unit
#[derive(Debug)]
pub struct GRU {
    flat_weights: Vec<Tensor>,
    hidden_dim: i64,
    config: RNNConfig,
    device: Device,
}

/// Creates a new GRU layer.
pub fn gru(vs: &super::var_store::Path, in_dim: i64, hidden_dim: i64, c: RNNConfig) -> GRU {
    let num_directions = if c.bidirectional { 2 } else { 1 };
    let gate_dim = 3 * hidden_dim;
    let mut flat_weights = vec![];
    for layer_idx in 0..c.num_layers {
        for _direction_idx in 0..num_directions {
            let in_dim = if layer_idx == 0 {
                in_dim
            } else {
                hidden_dim * num_directions
            };
            let w_ih = vs.kaiming_uniform("w_ih", &[gate_dim, in_dim]);
            let w_hh = vs.kaiming_uniform("w_hh", &[gate_dim, hidden_dim]);
            let b_ih = vs.zeros("b_ih", &[gate_dim]);
            let b_hh = vs.zeros("b_hh", &[gate_dim]);
            flat_weights.push(w_ih);
            flat_weights.push(w_hh);
            flat_weights.push(b_ih);
            flat_weights.push(b_hh);
        }
    }
    GRU {
        flat_weights,
        hidden_dim,
        config: c,
        device: vs.device(),
    }
}

impl RNN for GRU {
    type State = GRUState;

    fn zero_state(&self, batch_dim: i64) -> GRUState {
        let num_directions = if self.config.bidirectional { 2 } else { 1 };
        let layer_dim = self.config.num_layers * num_directions;
        let shape = [layer_dim, batch_dim, self.hidden_dim];
        GRUState(Tensor::zeros(&shape, (Kind::Float, self.device)))
    }

    fn step(&self, input: &Tensor, in_state: &GRUState) -> GRUState {
        let input = input.unsqueeze(1);
        let (_output, state) = self.seq_init(&input, in_state);
        state
    }

    fn seq_init(&self, input: &Tensor, in_state: &GRUState) -> (Tensor, GRUState) {
        let GRUState(h) = in_state;
        let (output, h) = input.gru(
            h,
            &self.flat_weights,
            self.config.has_biases,
            self.config.num_layers,
            self.config.dropout,
            self.config.train,
            self.config.bidirectional,
            self.config.batch_first,
        );
        (output, GRUState(h))
    }
}

impl ToDevice for GRU {
    fn to_device(&self, device: Device) -> Self {
        Self {
            flat_weights: self
                .flat_weights
                .iter()
                .map(|x| x.to_device(device))
                .collect::<Vec<_>>(),
            hidden_dim: self.hidden_dim,
            config: self.config,
            device,
        }
    }
}
