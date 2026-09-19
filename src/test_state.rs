use bc_utils_lg::prelude::*;

pub static SIGNALS_TRAIN_STATE: LazyLock<MAP<&str, f64>> =
    LazyLock::new(|| MAP::from_iter([("mm_1", 0.)]));
