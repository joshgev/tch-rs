use tch::nn::{Module, OptimizerConfig};
use tch::{kind, nn, Device, Kind, Reduction, Tensor};

#[test]
fn optimizer_test() {
    tch::manual_seed(42);
    // Create some linear data.
    let xs = Tensor::of_slice(&(1..15).collect::<Vec<_>>())
        .to_kind(Kind::Float)
        .view([-1, 1]);
    let ys = &xs * 0.42 + 1.337;

    // Fit a linear model (with deterministic initialization) on the data.
    let vs = nn::VarStore::new(Device::Cpu);
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
    let cfg = nn::LinearConfig {
        ws_init: nn::Init::Const(0.),
        bs_init: Some(nn::Init::Const(0.)),
    };
    let mut linear = nn::linear(vs.root(), 1, 1, cfg);

    let loss = xs.apply(&linear).mse_loss(&ys, Reduction::Mean);
    let initial_loss = f64::from(&loss);
    assert!(initial_loss > 1.0, "initial loss {}", initial_loss);

    // Optimization loop.
    for _idx in 1..50 {
        let loss = xs.apply(&linear).mse_loss(&ys, Reduction::Mean);
        opt.backward_step(&loss);
    }
    let loss = xs.apply(&linear).mse_loss(&ys, Reduction::Mean);
    let final_loss = f64::from(loss);
    assert!(final_loss < 0.25, "final loss {}", final_loss);

    // Reset the weights to their initial values.
    tch::no_grad(|| {
        linear.ws.init(nn::Init::Const(0.));
        linear.bs.init(nn::Init::Const(0.));
    });
    let initial_loss2 = f64::from(xs.apply(&linear).mse_loss(&ys, Reduction::Mean));
    assert_eq!(initial_loss, initial_loss2)
}

fn my_module(p: nn::Path, dim: i64) -> impl nn::Module {
    let x1 = p.zeros("x1", &[dim]);
    let x2 = p.zeros("x2", &[dim]);
    nn::func(move |xs| xs * &x1 + xs.exp() * &x2)
}

#[test]
fn gradient_descent_test() {
    let vs = nn::VarStore::new(Device::Cpu);
    let my_module = my_module(vs.root(), 7);
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
    for _idx in 1..50 {
        let xs = Tensor::zeros(&[7], kind::FLOAT_CPU);
        let ys = Tensor::zeros(&[7], kind::FLOAT_CPU);
        let loss = (my_module.forward(&xs) - ys).pow(2).sum(Kind::Float);
        opt.backward_step(&loss);
    }
}

#[test]
fn bn_test() {
    let opts = (tch::Kind::Float, tch::Device::Cpu);
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let bn = nn::batch_norm1d(vs.root(), 40, Default::default());
    let x = Tensor::randn(&[10, 40], opts);
    let _y = x.apply_t(&bn, true);
}

fn gru_test(rnn_config: nn::RNNConfig) {
    use nn::RNN;
    let batch_dim = 5;
    let seq_len = 3;
    let input_dim = 2;
    let output_dim = 4;
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let gru = nn::gru(&vs.root(), input_dim, output_dim, rnn_config);

    let num_directions = if rnn_config.bidirectional { 2 } else { 1 };
    let layer_dim = rnn_config.num_layers * num_directions;
    //
    // step test
    let input = Tensor::randn(&[batch_dim, input_dim], kind::FLOAT_CPU);
    let nn::GRUState(output) = gru.step(&input, &gru.zero_state(batch_dim));
    assert_eq!(output.size(), [layer_dim, batch_dim, output_dim]);

    // seq test
    let input = Tensor::randn(&[batch_dim, seq_len, input_dim], kind::FLOAT_CPU);
    let (output, _) = gru.seq(&input);
    assert_eq!(
        output.size(),
        [batch_dim, seq_len, output_dim * num_directions]
    );
}

#[test]
fn gru() {
    gru_test(Default::default());
    gru_test(nn::RNNConfig {
        bidirectional: true,
        ..Default::default()
    });
    gru_test(nn::RNNConfig {
        num_layers: 2,
        ..Default::default()
    });
    gru_test(nn::RNNConfig {
        num_layers: 2,
        bidirectional: true,
        ..Default::default()
    });
}

fn lstm_test(rnn_config: nn::RNNConfig) {
    use nn::RNN;
    let batch_dim = 5;
    let seq_len = 3;
    let input_dim = 2;
    let output_dim = 4;
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let lstm = nn::lstm(&vs.root(), input_dim, output_dim, rnn_config);

    let num_directions = if rnn_config.bidirectional { 2 } else { 1 };
    let layer_dim = rnn_config.num_layers * num_directions;
    //
    // step test
    let input = Tensor::randn(&[batch_dim, input_dim], kind::FLOAT_CPU);
    let nn::LSTMState((h, c)) = lstm.step(&input, &lstm.zero_state(batch_dim));
    assert_eq!(h.size(), [layer_dim, batch_dim, output_dim]);
    assert_eq!(c.size(), [layer_dim, batch_dim, output_dim]);

    // seq test
    let input = Tensor::randn(&[batch_dim, seq_len, input_dim], kind::FLOAT_CPU);
    let (output, _) = lstm.seq(&input);
    assert_eq!(
        output.size(),
        [batch_dim, seq_len, output_dim * num_directions]
    );
}

#[test]
fn lstm() {
    lstm_test(Default::default());
    lstm_test(nn::RNNConfig {
        bidirectional: true,
        ..Default::default()
    });
    lstm_test(nn::RNNConfig {
        num_layers: 2,
        ..Default::default()
    });
    lstm_test(nn::RNNConfig {
        num_layers: 2,
        bidirectional: true,
        ..Default::default()
    });
}

fn make_weights(
    ty: nn::RnnType,
    input_size: i64,
    hidden_size: i64,
    with_bias: bool,
) -> nn::RnnWeights {
    let gate_size = hidden_size * ty.num_gates();
    nn::RnnWeights::new(
        Tensor::randn(&[gate_size, input_size], (Kind::Float, Device::Cpu)),
        Tensor::randn(&[gate_size, hidden_size], (Kind::Float, Device::Cpu)),
        if with_bias {
            Some(Tensor::randn(&[gate_size], (Kind::Float, Device::Cpu)))
        } else {
            None
        },
        if with_bias {
            Some(Tensor::randn(&[gate_size], (Kind::Float, Device::Cpu)))
        } else {
            None
        },
    )
    .unwrap()
}

fn make_layer(
    ty: nn::RnnType,
    input_size: i64,
    hidden_size: i64,
    with_bias: bool,
    bidirectional: bool,
) -> nn::RnnLayer {
    nn::RnnLayer::new(
        make_weights(ty, input_size, hidden_size, with_bias),
        if bidirectional {
            Some(make_weights(ty, input_size, hidden_size, with_bias))
        } else {
            None
        },
    )
    .unwrap()
}

#[test]
fn builder_weights() {
    // An RNN will have weights of size:
    //   w_ih: (num_gates * hidden_size, input_size)
    //   w_hh: (num_gates * hidden_size, hidden_size)
    //   b_ih: (num_gates * hidden_size)
    //   b_hh: (num_gates * hidden_size)
    // where `input_size` is the previous layer's output size for the first
    // RNN layer, and `hidden_size * num_directions` for subsequent layers.

    const INPUT_SIZE: i64 = 3;
    const HIDDEN_SIZE: i64 = 7;

    for &rnn_type in &[nn::RnnType::Lstm, nn::RnnType::Gru] {
        for &is_first in &[false, true] {
            for &with_bias in &[false, true] {
                for &num_directions in &[1, 2] {
                    let input_size = if is_first {
                        INPUT_SIZE
                    } else {
                        HIDDEN_SIZE * num_directions
                    };

                    let x = make_weights(rnn_type, input_size, HIDDEN_SIZE, with_bias);

                    assert_eq!(x.rnn_type(), rnn_type);
                    assert_eq!(x.has_biases(), with_bias);
                    assert_eq!(x.input_features(), input_size);
                    assert_eq!(x.hidden_dim(), HIDDEN_SIZE);
                }
            }
        }
    }
}

#[test]
fn rnn_builder() {
    const INPUT_SIZE: i64 = 2;
    const HIDDEN_SIZE: i64 = 3;
    const BIDIRECTIONAL: bool = true;
    const WITH_BIAS: bool = true;

    let mut builder = nn::RnnBuilder::new(nn::RnnBuilderConfig::default());

    assert_eq!(builder.input_features(), None);
    assert_eq!(builder.hidden_dim(), None);
    assert_eq!(builder.output_features(), None);
    assert_eq!(builder.bidirectional(), None);
    assert_eq!(builder.has_biases(), None);
    assert_eq!(builder.num_layers(), 0);
    assert_eq!(builder.rnn_type(), None);
    assert!(builder.rnn_config().is_none());

    builder
        .push(make_layer(
            nn::RnnType::Gru,
            INPUT_SIZE,
            HIDDEN_SIZE,
            WITH_BIAS,
            BIDIRECTIONAL,
        ))
        .unwrap();

    assert_eq!(builder.input_features(), Some(INPUT_SIZE));
    assert_eq!(builder.hidden_dim(), Some(HIDDEN_SIZE));
    assert_eq!(
        builder.output_features(),
        Some(HIDDEN_SIZE * if BIDIRECTIONAL { 2 } else { 1 })
    );
    assert_eq!(builder.bidirectional(), Some(BIDIRECTIONAL));
    assert_eq!(builder.has_biases(), Some(WITH_BIAS));
    assert_eq!(builder.num_layers(), 1);
    assert_eq!(builder.rnn_type(), Some(nn::RnnType::Gru));
    let config = builder.rnn_config().unwrap();
    assert_eq!(config.has_biases, WITH_BIAS);
    assert_eq!(config.num_layers, 1);
    assert_eq!(config.bidirectional, BIDIRECTIONAL);

    builder
        .push(make_layer(
            nn::RnnType::Gru,
            HIDDEN_SIZE * if BIDIRECTIONAL { 2 } else { 1 },
            HIDDEN_SIZE,
            WITH_BIAS,
            BIDIRECTIONAL,
        ))
        .unwrap();

    assert_eq!(builder.input_features(), Some(INPUT_SIZE));
    assert_eq!(builder.hidden_dim(), Some(HIDDEN_SIZE));
    assert_eq!(
        builder.output_features(),
        Some(HIDDEN_SIZE * if BIDIRECTIONAL { 2 } else { 1 })
    );
    assert_eq!(builder.bidirectional(), Some(BIDIRECTIONAL));
    assert_eq!(builder.has_biases(), Some(WITH_BIAS));
    assert_eq!(builder.num_layers(), 2);
    assert_eq!(builder.rnn_type(), Some(nn::RnnType::Gru));
    let config = builder.rnn_config().unwrap();
    assert_eq!(config.has_biases, WITH_BIAS);
    assert_eq!(config.num_layers, 2);
    assert_eq!(config.bidirectional, BIDIRECTIONAL);

    assert!(builder
        .push(make_layer(nn::RnnType::Lstm, 6, 3, true, true))
        .is_err());
    assert!(builder
        .push(make_layer(nn::RnnType::Gru, 4, 3, true, true))
        .is_err());
    assert!(builder
        .push(make_layer(nn::RnnType::Gru, 6, 4, true, true))
        .is_err());
    assert!(builder
        .push(make_layer(nn::RnnType::Gru, 6, 3, false, true))
        .is_err());
    assert!(builder
        .push(make_layer(nn::RnnType::Gru, 6, 3, true, false))
        .is_err());

    let rnn = builder.gru().unwrap();

    const BATCH_SIZE: i64 = 5;
    const TIMESTEPS: i64 = 7;

    let x = Tensor::randn(
        &[BATCH_SIZE, TIMESTEPS, INPUT_SIZE],
        (Kind::Float, Device::Cpu),
    );

    use nn::RNN;
    let (y, h) = rnn.seq(&x);

    const NUM_LAYERS: i64 = 2;

    assert_eq!(
        y.size(),
        [
            BATCH_SIZE,
            TIMESTEPS,
            HIDDEN_SIZE * if BIDIRECTIONAL { 2 } else { 1 }
        ]
    );
    assert_eq!(
        h.0.size(),
        [
            NUM_LAYERS * if BIDIRECTIONAL { 2 } else { 1 },
            BATCH_SIZE,
            HIDDEN_SIZE
        ]
    );
}
