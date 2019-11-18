//! Recurrent Neural Networks
use crate::{AsView, Device, Kind, Tensor, ToDevice};
use failure::{bail, ensure, Error};

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

impl ToDevice for LSTMState {
    fn to_device(&self, device: Device) -> Self {
        Self(((self.0).0.to_device(device), (self.0).1.to_device(device)))
    }

    fn device(&self) -> Device {
        (self.0).0.device()
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

    fn device(&self) -> Device {
        self.device
    }
}

impl AsView for LSTM {
    fn as_view(&self) -> Self {
        Self {
            flat_weights: self
                .flat_weights
                .iter()
                .map(|x| x.as_view())
                .collect::<Vec<_>>(),
            hidden_dim: self.hidden_dim,
            config: self.config,
            device: self.device,
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

impl ToDevice for GRUState {
    fn to_device(&self, device: Device) -> Self {
        Self(self.0.to_device(device))
    }

    fn device(&self) -> Device {
        self.0.device()
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

    fn device(&self) -> Device {
        self.device
    }
}

impl AsView for GRU {
    fn as_view(&self) -> Self {
        Self {
            flat_weights: self
                .flat_weights
                .iter()
                .map(|x| x.as_view())
                .collect::<Vec<_>>(),
            hidden_dim: self.hidden_dim,
            config: self.config,
            device: self.device,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RnnType {
    Lstm,
    Gru,
}

impl RnnType {
    pub fn num_gates(self) -> i64 {
        match self {
            RnnType::Lstm => 4,
            RnnType::Gru => 3,
        }
    }

    fn from_num_gates(num_gates: i64) -> Result<Self, Error> {
        match num_gates {
            3 => Ok(RnnType::Gru),
            4 => Ok(RnnType::Lstm),
            _ => bail!("Unknown RNN type from number of gates: {}", num_gates),
        }
    }
}

pub struct RnnWeights {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Option<Tensor>,
    b_hh: Option<Tensor>,
    rnn_type: RnnType,
}

impl RnnWeights {
    pub fn new(
        w_ih: Tensor,
        w_hh: Tensor,
        b_ih: Option<Tensor>,
        b_hh: Option<Tensor>,
    ) -> Result<Self, Error> {
        ensure!(w_ih.dim() == 2, "w_ih should be 2D.");
        ensure!(w_hh.dim() == 2, "w_hh should be 2D.");

        ensure!(w_ih.device() == w_hh.device(), "Device mis-match.");

        let w_ih_size = w_ih.size2().unwrap();
        let w_hh_size = w_ih.size2().unwrap();

        ensure!(
            w_ih_size.0 == w_hh_size.0,
            "w_ih and w_hh are incompatible."
        );

        if let Some(ref b_ih) = b_ih {
            ensure!(b_ih.dim() == 1, "b_ih should be 1D.");
            let b_ih_size = b_ih.size1().unwrap();
            ensure!(w_ih_size.0 == b_ih_size, "w_ih and b_ih are incompatible.");
            ensure!(w_ih.device() == b_ih.device(), "Device mis-match.");
        }

        if let Some(ref b_hh) = b_hh {
            ensure!(b_hh.dim() == 1, "b_hh should be 1D.");
            let b_hh_size = b_hh.size1().unwrap();
            ensure!(w_hh_size.0 == b_hh_size, "w_hh and b_hh are incompatible.");
            ensure!(w_hh.device() == b_hh.device(), "Device mis-match.");
        }

        ensure!(
            b_ih.is_some() == b_hh.is_some(),
            "Either both b_ih, b_hh must be specified, or neither."
        );

        let rnn_type = RnnType::from_num_gates({
            let size = w_hh.size2().unwrap();
            size.0 / size.1
        })?;

        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            rnn_type,
        })
    }

    pub fn rnn_type(&self) -> RnnType {
        self.rnn_type
    }

    pub fn has_biases(&self) -> bool {
        self.b_ih.is_some()
    }

    pub fn input_features(&self) -> i64 {
        self.w_ih.size2().unwrap().1
    }

    pub fn hidden_dim(&self) -> i64 {
        self.w_ih.size2().unwrap().0 / self.rnn_type().num_gates()
    }

    pub fn device(&self) -> Device {
        self.w_ih.device()
    }

    fn push_into(self, dst: &mut Vec<Tensor>) {
        dst.push(self.w_ih);
        dst.push(self.w_hh);
        if let Some(b_ih) = self.b_ih {
            dst.push(b_ih);
            dst.push(self.b_hh.expect("Inconsistent bias. This is a bug."));
        }
    }

    #[must_use]
    pub fn set_requires_grad(&self, requires_grad: bool) -> Self {
        Self {
            w_ih: self.w_ih.set_requires_grad(requires_grad),
            w_hh: self.w_hh.set_requires_grad(requires_grad),
            b_ih: self
                .b_ih
                .as_ref()
                .map(|x| x.set_requires_grad(requires_grad)),
            b_hh: self
                .b_hh
                .as_ref()
                .map(|x| x.set_requires_grad(requires_grad)),
            rnn_type: self.rnn_type,
        }
    }
}

pub struct RnnLayer {
    forward: RnnWeights,
    reverse: Option<RnnWeights>,
}

impl RnnLayer {
    pub fn new(forward: RnnWeights, reverse: Option<RnnWeights>) -> Result<Self, Error> {
        if let Some(ref reverse) = reverse {
            ensure!(
                forward.has_biases() == reverse.has_biases(),
                "Forward and reverse layers have different bias settings."
            );
            ensure!(
                forward.input_features() == reverse.input_features(),
                "Forward and reverse layers have different numbers of input feautres."
            );
            ensure!(
                forward.hidden_dim() == reverse.hidden_dim(),
                "Forward and reverse layers have different hidden dimensionality."
            );
            ensure!(forward.device() == reverse.device(), "Device mis-match.");
            ensure!(
                forward.rnn_type() == reverse.rnn_type(),
                "RNN type mis-match."
            );
        }

        Ok(Self { forward, reverse })
    }

    pub fn has_biases(&self) -> bool {
        self.forward.has_biases()
    }

    pub fn input_features(&self) -> i64 {
        self.forward.input_features()
    }

    pub fn bidirectional(&self) -> bool {
        self.reverse.is_some()
    }

    pub fn hidden_dim(&self) -> i64 {
        self.forward.hidden_dim()
    }

    pub fn output_features(&self) -> i64 {
        self.forward.hidden_dim() * if self.bidirectional() { 2 } else { 1 }
    }

    pub fn rnn_type(&self) -> RnnType {
        self.forward.rnn_type()
    }

    pub fn device(&self) -> Device {
        self.forward.device()
    }

    fn push_into(self, dst: &mut Vec<Tensor>) {
        self.forward.push_into(dst);
        if let Some(reverse) = self.reverse {
            reverse.push_into(dst);
        }
    }

    #[must_use]
    pub fn set_requires_grad(&self, requires_grad: bool) -> Self {
        Self {
            forward: self.forward.set_requires_grad(requires_grad),
            reverse: self
                .reverse
                .as_ref()
                .map(|x| x.set_requires_grad(requires_grad)),
        }
    }
}

pub struct RnnBuilder {
    layers: Vec<RnnLayer>,
    config: RnnBuilderConfig,
}

#[derive(Debug, Copy, Clone)]
pub struct RnnBuilderConfig {
    pub dropout: f64,
    pub train: bool,
    pub batch_first: bool,
}

impl Default for RnnBuilderConfig {
    fn default() -> Self {
        let x = RNNConfig::default();
        Self {
            dropout: x.dropout,
            train: x.train,
            batch_first: x.batch_first,
        }
    }
}

impl RnnBuilder {
    pub fn new(config: RnnBuilderConfig) -> Self {
        Self {
            layers: Default::default(),
            config,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn input_features(&self) -> Option<i64> {
        self.layers.first().map(|layer| layer.input_features())
    }

    pub fn hidden_dim(&self) -> Option<i64> {
        self.layers.last().map(|layer| layer.hidden_dim())
    }

    pub fn output_features(&self) -> Option<i64> {
        self.layers.last().map(|layer| layer.output_features())
    }

    pub fn bidirectional(&self) -> Option<bool> {
        self.layers.first().map(|layer| layer.bidirectional())
    }

    pub fn has_biases(&self) -> Option<bool> {
        self.layers.first().map(|layer| layer.has_biases())
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len() as usize
    }

    pub fn len(&self) -> usize {
        self.num_layers()
    }

    pub fn rnn_type(&self) -> Option<RnnType> {
        self.layers.first().map(|layer| layer.rnn_type())
    }

    pub fn device(&self) -> Option<Device> {
        self.layers.first().map(|layer| layer.device())
    }

    pub fn rnn_config(&self) -> Option<RNNConfig> {
        if self.layers.is_empty() {
            None
        } else {
            Some(RNNConfig {
                has_biases: self.has_biases().unwrap(),
                num_layers: self.num_layers() as i64,
                dropout: self.config.dropout,
                train: self.config.train,
                bidirectional: self.bidirectional().unwrap(),
                batch_first: self.config.batch_first,
            })
        }
    }

    pub fn push(&mut self, layer: RnnLayer) -> Result<&mut Self, Error> {
        if let Some(last) = self.layers.last() {
            ensure!(
                last.output_features() == layer.input_features(),
                "Input/output feature dimensionality mismatch."
            );
            ensure!(
                last.output_features() == layer.output_features(),
                "Inconsistent hidden state size."
            );
            ensure!(last.has_biases() == layer.has_biases(), "Change in biases.");
            ensure!(
                last.bidirectional() == layer.bidirectional(),
                "Change in bidirectionality."
            );
            ensure!(last.rnn_type() == layer.rnn_type(), "Change in RNN type.");
            ensure!(last.device() == layer.device(), "Device mis-match.");
        }

        self.layers.push(layer);

        Ok(self)
    }

    fn gather_weights(self) -> Vec<Tensor> {
        let mut weights = Vec::with_capacity(
            self.num_layers()
                * if self.bidirectional().unwrap() { 2 } else { 1 }
                * if self.has_biases().unwrap() { 2 } else { 1 },
        );

        for layer in self.layers {
            layer.push_into(&mut weights);
        }

        weights
    }

    pub fn lstm(self) -> Result<LSTM, Error> {
        ensure!(!self.layers.is_empty(), "No layers defined.");
        ensure!(
            self.rnn_type().unwrap() == RnnType::Lstm,
            "Underlying RNN is not an LSTM."
        );

        Ok(LSTM {
            hidden_dim: self.hidden_dim().unwrap(),
            config: self.rnn_config().unwrap(),
            device: self.device().unwrap(),
            flat_weights: self.gather_weights(),
        })
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) -> &mut Self {
        for layer in &mut self.layers {
            *layer = layer.set_requires_grad(requires_grad);
        }
        self
    }

    pub fn gru(self) -> Result<GRU, Error> {
        ensure!(!self.layers.is_empty(), "No layers defined.");
        ensure!(
            self.rnn_type().unwrap() == RnnType::Gru,
            "Underlying RNN is not a GRU."
        );

        Ok(GRU {
            hidden_dim: self.hidden_dim().unwrap(),
            config: self.rnn_config().unwrap(),
            device: self.device().unwrap(),
            flat_weights: self.gather_weights(),
        })
    }
}
