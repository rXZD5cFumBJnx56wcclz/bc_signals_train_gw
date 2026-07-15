use std::hint::black_box;

use bc_pack_signals_train::FUNCS_EXTRACT_ARGS as FUNCS_EXTRACT_ARGS_ST;
use bc_utils_lg::statics::prices::SRC_TRANSPOSE;
use bc_utils_lg::structs::settings::{SETTINGS_SIGNAL, SETTINGS_SIGNALS, SETTINGS_USED_USIZE};
use bc_utils_lg::types::maps::MAP;
use criterion::{Criterion, criterion_group, criterion_main};

use bc_signals_train_gw::gw::{SignalsTrain, SignalsTrainGateway};

fn get_signals_train_from_settings_1(c: &mut Criterion) {
    let settings_signals = SETTINGS_SIGNALS::from_iter([(
        "mm_1".to_string(),
        SETTINGS_SIGNAL {
            key: "mm".to_string(),
            kwargs_usize: MAP::from_iter([("window".to_string(), 10)]),
            used_src: vec![SETTINGS_USED_USIZE { index: 1, sub_from_last_i: 0 }],
            ..Default::default()
        },
    )]);
    let bind = Default::default();
    let bind2 = Default::default();
    let bind3 = Default::default();
    let bind4 = Default::default();
    let bind5 = Default::default();
    let sr = SignalsTrain::new(
        &settings_signals,
        &bind,
        &FUNCS_EXTRACT_ARGS_ST(),
        &SRC_TRANSPOSE,
        &bind2,
    );
    let sr_gw = SignalsTrainGateway::new(&sr, &bind3, &settings_signals, &bind4);
    c.bench_function("get_signals_train_from_settings_1", |b| {
        b.iter(|| sr_gw.signals_series(black_box(&bind5), black_box(&SRC_TRANSPOSE)))
    });
}

criterion_group!(benches, get_signals_train_from_settings_1,);
criterion_main!(benches);
