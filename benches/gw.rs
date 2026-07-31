use std::hint::black_box;

use bc_indicators_gw::gw::Indicators;
use bc_packs::{PACK_IND, PACK_SIGN_TR};
use bc_test_kit::prelude::*;
use bc_test_kit::settings::signals_train::SIGNALS_TRAIN;
use criterion::{Criterion, criterion_group, criterion_main};

use bc_signals_train_gw::gw::SignalsTrain;

fn get_signals_train_from_settings_1(c: &mut Criterion) {
    let indicators = Indicators::new(&SRC_TRANSPOSE, &INDICATIONS, &PACK_IND);
    let indications = indicators.series(&SRC_TRANSPOSE, &INDICATIONS);
    let sr = SignalsTrain::new(
        &SRC_TRANSPOSE,
        &SIGNALS_TRAIN,
        &INDICATIONS,
        &indicators,
        &PACK_SIGN_TR,
    );
    c.bench_function("get_signals_train_from_settings_1", |b| {
        b.iter(|| {
            sr.series(
                black_box(&SRC_TRANSPOSE),
                black_box(&SIGNALS_TRAIN),
                black_box(&indications),
            )
        })
    });
}

criterion_group!(benches, get_signals_train_from_settings_1,);
criterion_main!(benches);
