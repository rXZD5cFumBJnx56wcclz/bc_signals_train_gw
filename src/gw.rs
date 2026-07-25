use bc_indicators::prelude::Indicator;
use bc_signals_train::prelude::*;
use bc_utils::other::{procedure_used, transpose, vec_len_sync_set};
use bc_utils_lg::structs::settings::{SETTINGS_INDS, SETTINGS_SIGNAL, SETTINGS_SIGNALS};
use bc_utils_lg::{
    structs::settings::SETTINGS,
    types::maps::{MAP, PACK},
};

use bc_indicators_gw::gw::{Indicators, get_src as get_src_ind};

pub fn get_src<'a>(
    s: &SETTINGS_SIGNAL,
    s_inds: &SETTINGS_INDS,
    s_signals_train: &SETTINGS_SIGNALS,
    src_transpose: &[Vec<f64>],
    map_ind_without_bf: &MAP<&'a str, Box<dyn Indicator>>,
    map_signals_train_without_bf: &MAP<&'a str, Box<dyn SignalTrain>>,
) -> Vec<Vec<f64>> {
    let mut res = vec![];
    for used_src_el in &s.used_src {
        res.push({
            let sk = &src_transpose[used_src_el.index];
            sk[..sk.len() - used_src_el.sub_from_last_i].to_vec()
        });
    }
    for used_ind_el in &s.used_ind {
        res.push(
            map_ind_without_bf[used_ind_el.as_str()].ind_vec(&get_src_ind(
                &s_inds[used_ind_el.as_str()],
                s_inds,
                src_transpose,
                map_ind_without_bf,
            )),
        );
    }
    for used_signals_train in &s.used_signals_train {
        res.push(
            map_signals_train_without_bf[used_signals_train.as_str()].signals_vec(&get_src(
                &s_signals_train[used_signals_train.as_str()],
                s_inds,
                s_signals_train,
                src_transpose,
                map_ind_without_bf,
                map_signals_train_without_bf,
            )),
        );
    }
    if !s.procedure_used_src.is_empty() {
        res = procedure_used(res, &s.procedure_used_src);
    }
    if !res.is_empty() {
        vec_len_sync_set(&mut res);
        return transpose(res);
    }
    Default::default()
}

fn get_src_series(
    s: &SETTINGS_SIGNAL,
    src_transpose: &[Vec<f64>],
    indications: &MAP<&str, f64>,
    signals_train: &MAP<&str, f64>,
) -> Vec<f64> {
    let mut res = vec![];
    for src_arg_el in &s.used_src {
        res.push({
            let sk = &src_transpose[src_arg_el.index];
            sk[sk.len() - 1 - src_arg_el.sub_from_last_i]
        });
    }
    for ind_arg_el in &s.used_ind {
        res.push(indications[ind_arg_el.as_str()]);
    }
    for signals_arg_el in &s.used_signals {
        res.push(signals_train[signals_arg_el.as_str()].clone());
    }
    if !s.procedure_used_src.is_empty() {
        res = procedure_used(res, &s.procedure_used_src);
    }
    res
}

pub fn get_map_from_pack<'a>(
    settings: &'a SETTINGS_SIGNALS,
    pack: &PACK<SETTINGS_SIGNAL, Box<dyn SignalTrain>>,
) -> MAP<&'a str, Box<dyn SignalTrain>> {
    settings
        .iter()
        .map(|(signal_name, settings_signal)| {
            let signal = pack[settings_signal.key.as_str()](settings_signal);
            (signal_name.as_str(), signal)
        })
        .collect()
}

pub fn get_map<'a>(
    settings_signals: &'a SETTINGS_SIGNALS,
    settings_indicators: &'a SETTINGS_INDS,
    pack: &MAP<&'a str, fn(&SETTINGS_SIGNAL) -> Box<dyn SignalTrain>>,
    src_transpose: &[Vec<f64>],
    map_signals: &MAP<&'a str, Box<dyn SignalTrain>>,
    map_indicators: &MAP<&'a str, Box<dyn Indicator>>,
) -> MAP<&'a str, Box<dyn SignalTrain>> {
    settings_signals
        .iter()
        .map(|(signal_name, settings_signal)| {
            let signal = pack[settings_signal.key.as_str()](settings_signal);
            signal.init_bf(&get_src(
                settings_signal,
                settings_indicators,
                settings_signals,
                src_transpose,
                map_indicators,
                map_signals,
            ));
            (signal_name.as_str(), signal)
        })
        .collect()
}

#[derive(Default)]
pub struct SignalsTrain<'a> {
    pub signals_train_without_bf: MAP<&'a str, Box<dyn SignalTrain>>,
    pub signals_train: MAP<&'a str, Box<dyn SignalTrain>>,
}

impl<'a> SignalsTrain<'a> {
    pub fn new(
        s_signals_train: &'a SETTINGS_SIGNALS,
        s_indicators: &'a SETTINGS_INDS,
        pack: &PACK<SETTINGS_SIGNAL, Box<dyn SignalTrain>>,
        src_transpose: &[Vec<f64>],
        map_indicators: &MAP<&'a str, Box<dyn Indicator>>,
    ) -> Self {
        let signals_train_without_bf = get_map_from_pack(s_signals_train, pack);
        Self {
            signals_train: get_map(
                s_signals_train,
                s_indicators,
                pack,
                src_transpose,
                &signals_train_without_bf,
                map_indicators,
            ),
            signals_train_without_bf,
        }
    }
    pub fn update_bf<'b>(
        &mut self,
        src_transpose: &[Vec<f64>],
        s: &'a SETTINGS,
        fa: &'b PACK<SETTINGS_SIGNAL, Box<dyn SignalTrain>>,
        indicators_without_bf: &MAP<&'a str, Box<dyn Indicator>>,
    ) {
        self.signals_train = get_map(
            &s.pipeline.signals_train,
            &s.pipeline.indications,
            fa,
            src_transpose,
            &self.signals_train_without_bf,
            indicators_without_bf,
        );
    }
}

#[derive(Default)]
pub struct SignalsTrainGateway<'a> {
    pub signals_train: *const SignalsTrain<'a>,
    pub indicators: *const Indicators<'a>,
    pub settings_signals: *const SETTINGS_SIGNALS,
    pub settings_indicators: *const SETTINGS_INDS,
}

impl<'a> SignalsTrainGateway<'a> {
    pub fn new(
        signals_train: *const SignalsTrain<'a>,
        indicators: *const Indicators<'a>,
        settings_signals: *const SETTINGS_SIGNALS,
        settings_indicators: *const SETTINGS_INDS,
    ) -> Self {
        Self {
            signals_train,
            indicators,
            settings_signals,
            settings_indicators,
        }
    }
}

impl<'a> SignalsTrainGateway<'a> {
    pub fn signals_series(
        &self,
        indications: &MAP<&str, f64>,
        src_transpose: &[Vec<f64>],
    ) -> MAP<&'a str, f64> {
        unsafe { &*self.settings_signals }
            .iter()
            .fold(MAP::default(), |mut map, setting| {
                let key_uniq_str = setting.0.as_str();
                let signal = unsafe { &(&(*self.signals_train).signals_train)[key_uniq_str] };
                map.insert(
                    key_uniq_str,
                    signal.signal_with_bf(&get_src_series(
                        &setting.1,
                        src_transpose,
                        indications,
                        &map,
                    )),
                );
                map
            })
    }
    pub fn signals_vec(&self, src_transpose: &[Vec<f64>]) -> MAP<&'a str, Vec<f64>> {
        unsafe { &*self.settings_signals }
            .iter()
            .map(|(k, setting)| {
                let key_uniq = k.as_str();
                let signal = unsafe { &(&(*self.signals_train).signals_train)[key_uniq] };
                (
                    key_uniq,
                    signal.signals_vec(&get_src(
                        setting,
                        unsafe { &*self.settings_indicators },
                        unsafe { &*self.settings_signals },
                        src_transpose,
                        &unsafe { &*self.indicators }.indicators_without_bf,
                        &unsafe { &*self.signals_train }.signals_train_without_bf,
                    )),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;

    use bc_indicators::trend_ma::TREND_MA;
    use bc_packs::{PACK_IND, PACK_SIGN_TR};
    use bc_signals_train::mm::MM;
    use bc_test_kit::prelude::*;
    use bc_utils_lg::structs::settings::{
        SETTINGS_IND, SETTINGS_INDS, SETTINGS_SIGNAL, SETTINGS_SIGNALS, SETTINGS_USED_USIZE,
    };
    use bc_utils_lg::types::maps::MAP;
    use pretty_assertions::assert_eq as assert_eq_pr;

    use bc_indicators_gw::gw::IndicatorsGateway;

    #[test]
    fn signals_from_settings_without_bf_res_1() {
        let settings = SETTINGS_SIGNALS::from_iter([(
            "mm_1".to_string(),
            SETTINGS_SIGNAL {
                key: "mm".to_string(),
                ..Default::default()
            },
        )]);
        let res = get_map_from_pack(&settings, &PACK_SIGN_TR);
        let res_1 = res.get("mm_1").unwrap().as_ref();
        let rsi_test_1 = MM::default();
        let rsi_test_2 = (res_1 as &dyn Any).downcast_ref::<MM>().unwrap();
        assert_eq_pr!(&rsi_test_1, rsi_test_2);
    }

    #[test]
    fn signals_train_res_1() {
        let settings_indicators = SETTINGS_INDS::from_iter([
            (
                "trend_ma_1".to_string(),
                SETTINGS_IND {
                    key: "trend_ma".to_string(),
                    used_src: vec![SETTINGS_USED_USIZE {
                        index: 1,
                        sub_from_last_i: 0,
                    }],
                    ..Default::default()
                },
            ),
            (
                "repeat_1".to_string(),
                SETTINGS_IND {
                    key: "repeat".to_string(),
                    kwargs_f64: MAP::from_iter([("value".to_string(), 1.0)]),
                    used_src: vec![SETTINGS_USED_USIZE {
                        index: 1,
                        sub_from_last_i: 0,
                    }],
                    ..Default::default()
                },
            ),
        ]);
        let settings_signals = SETTINGS_SIGNALS::from_iter([(
            "mm_1".to_string(),
            SETTINGS_SIGNAL {
                key: "mm".to_string(),
                kwargs_usize: MAP::from_iter([("window".to_string(), 10)]),
                used_ind: vec!["trend_ma_1".to_string(), "repeat_1".to_string()],
                ..Default::default()
            },
        )]);
        let indicators = Indicators::new(&settings_indicators, &PACK_IND, &SRC_TRANSPOSE);
        let signals = SignalsTrain::new(
            &settings_signals,
            &settings_indicators,
            &PACK_SIGN_TR,
            &SRC_TRANSPOSE,
            &indicators.indicators_without_bf,
        );
        let indicators_gw = IndicatorsGateway::new(&indicators, &settings_indicators);
        let indications = indicators_gw.indications_series(&SRC_TRANSPOSE);
        let signals_gw = SignalsTrainGateway::new(
            &signals,
            &indicators,
            &settings_signals,
            &settings_indicators,
        );
        let res_1 = signals_gw.signals_series(&indications, &SRC_TRANSPOSE)["mm_1"];
        let res_2 = {
            let mut df = MM::default();
            df.params.window = 10;
            df
        }
        .signal(
            &TREND_MA::default()
                .ind_vec(&SRC)
                .into_iter()
                .map(|v| vec![v])
                .collect::<Vec<Vec<f64>>>(),
        );
        assert_eq_pr!(res_1, res_2);
    }

    #[test]
    fn signals_train_vec_res_1() {
        let settings_indicators = SETTINGS_INDS::from_iter([
            (
                "trend_ma_1".to_string(),
                SETTINGS_IND {
                    key: "trend_ma".to_string(),
                    used_src: vec![SETTINGS_USED_USIZE {
                        index: 1,
                        sub_from_last_i: 0,
                    }],
                    ..Default::default()
                },
            ),
            (
                "repeat_1".to_string(),
                SETTINGS_IND {
                    key: "repeat".to_string(),
                    kwargs_f64: MAP::from_iter([("value".to_string(), 1.0)]),
                    used_src: vec![SETTINGS_USED_USIZE {
                        index: 1,
                        sub_from_last_i: 0,
                    }],
                    ..Default::default()
                },
            ),
        ]);
        let settings_signals = SETTINGS_SIGNALS::from_iter([(
            "mm_1".to_string(),
            SETTINGS_SIGNAL {
                key: "mm".to_string(),
                kwargs_usize: MAP::from_iter([("window".to_string(), 10)]),
                used_ind: vec!["trend_ma_1".to_string(), "repeat_1".to_string()],
                ..Default::default()
            },
        )]);
        let indicators = Indicators::new(&settings_indicators, &PACK_IND, &SRC_TRANSPOSE);
        let signals = SignalsTrain::new(
            &settings_signals,
            &settings_indicators,
            &PACK_SIGN_TR,
            &SRC_TRANSPOSE,
            &indicators.indicators_without_bf,
        );
        let signals_gw = SignalsTrainGateway::new(
            &signals,
            &indicators,
            &settings_signals,
            &settings_indicators,
        );
        let res_1 = &signals_gw.signals_vec(&SRC_TRANSPOSE)["mm_1"];
        let res_2 = &{
            let mut df = MM::default();
            df.params.window = 10;
            df
        }
        .signals_vec(
            &TREND_MA::default()
                .ind_vec(&SRC)
                .into_iter()
                .map(|v| vec![v])
                .collect::<Vec<Vec<f64>>>(),
        );
        assert_eq_pr!(
            res_1.iter().filter(|v| !v.is_nan()).collect::<Vec<&f64>>(),
            res_2.iter().filter(|v| !v.is_nan()).collect::<Vec<&f64>>()
        );
    }
}
