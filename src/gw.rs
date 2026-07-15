use bc_indicators::prelude::Indicator;
use bc_signals_train::prelude::*;
use bc_utils::other::{procedure_used, transpose, vec_len_sync_set};
use bc_utils_lg::structs::settings::{SETTINGS_INDS, SETTINGS_SIGNAL, SETTINGS_SIGNALS};
use bc_utils_lg::{
    structs::settings::SETTINGS,
    types::maps::{FUNCS_EXTRACT_ARGS_TYPE, MAP},
};

use bc_indicators_gw::gw::{Indicators, get_in_from_settings};

pub fn get_signals_arg_from_settings<'a>(
    used_signals: &Vec<String>,
    procedure_used_signals: &Vec<usize>,
    settings_signals: &SETTINGS_SIGNALS,
    settings_indicators: &SETTINGS_INDS,
    src_transpose: &[Vec<f64>],
    map_signals: &MAP<&'a str, Box<dyn SignalTrain>>,
    map_indicators: &MAP<&'a str, Box<dyn Indicator>>,
) -> Vec<Vec<f64>> {
    let mut res = vec![];
    for used_signal in used_signals {
        res.push(map_signals[used_signal.as_str()].signals_vec(
            &get_in_from_settings(
                &settings_signals[used_signal].used_ind,
                &settings_signals[used_signal].used_src,
                &settings_signals[used_signal].procedure_used_src,
                settings_indicators,
                src_transpose,
                map_indicators,
            ),
            &get_signals_arg_from_settings(
                &settings_signals[used_signal].used_signals,
                &settings_signals[used_signal].procedure_used_signals,
                settings_signals,
                settings_indicators,
                src_transpose,
                map_signals,
                map_indicators,
            ),
        ));
    }
    if !procedure_used_signals.is_empty() {
        res = procedure_used(res, procedure_used_signals)
    }
    if !res.is_empty() {
        vec_len_sync_set(&mut res);
        return transpose(res);
    }
    Default::default()
}

pub fn get_signals_from_settings_without_bf<'a>(
    settings: &'a SETTINGS_SIGNALS,
    funcs_extract_args: &MAP<&'a str, fn(&SETTINGS_SIGNAL) -> Box<dyn SignalTrain>>,
) -> MAP<&'a str, Box<dyn SignalTrain>> {
    settings
        .iter()
        .map(|(signal_name, settings_signal)| {
            let signal = funcs_extract_args[settings_signal.key.as_str()](settings_signal);
            (signal_name.as_str(), signal)
        })
        .collect()
}

pub fn get_signals_from_settings<'a>(
    settings_signals: &'a SETTINGS_SIGNALS,
    settings_indicators: &'a SETTINGS_INDS,
    funcs_extract_args: &MAP<&'a str, fn(&SETTINGS_SIGNAL) -> Box<dyn SignalTrain>>,
    src_transpose: &[Vec<f64>],
    map_signals: &MAP<&'a str, Box<dyn SignalTrain>>,
    map_indicators: &MAP<&'a str, Box<dyn Indicator>>,
) -> MAP<&'a str, (BF_SIGNALS<'a>, Box<dyn SignalTrain>)> {
    settings_signals
        .iter()
        .map(|(signal_name, settings_signal)| {
            let signal = funcs_extract_args[settings_signal.key.as_str()](settings_signal);
            let src_transpose = &src_transpose
                .into_iter()
                .map(|v| v[..v.len() - 1].to_vec())
                .collect::<Vec<Vec<f64>>>();
            (
                signal_name.as_str(),
                (
                    signal.bf(
                        &get_in_from_settings(
                            &settings_signal.used_ind,
                            &settings_signal.used_src,
                            &settings_signal.procedure_used_src,
                            settings_indicators,
                            src_transpose,
                            map_indicators,
                        ),
                        &get_signals_arg_from_settings(
                            &settings_signal.used_signals,
                            &settings_signal.procedure_used_signals,
                            settings_signals,
                            settings_indicators,
                            src_transpose,
                            map_signals,
                            map_indicators,
                        ),
                    ),
                    signal,
                ),
            )
        })
        .collect()
}

#[derive(Default)]
pub struct SignalsTrain<'a> {
    pub signals_train_without_bf: MAP<&'a str, Box<dyn SignalTrain>>,
    pub signals_train: MAP<&'a str, (BF_SIGNALS<'a>, Box<dyn SignalTrain>)>,
}

impl<'a> SignalsTrain<'a> {
    pub fn new(
        s_signals_train: &'a SETTINGS_SIGNALS,
        s_indicators: &'a SETTINGS_INDS,
        funcs_extract_args: &MAP<&'a str, fn(&SETTINGS_SIGNAL) -> Box<dyn SignalTrain>>,
        src_transpose: &[Vec<f64>],
        map_indicators: &MAP<&'a str, Box<dyn Indicator>>,
    ) -> Self {
        let signals_train_without_bf =
            get_signals_from_settings_without_bf(s_signals_train, funcs_extract_args);
        Self {
            signals_train: get_signals_from_settings(
                s_signals_train,
                s_indicators,
                funcs_extract_args,
                src_transpose,
                &signals_train_without_bf,
                map_indicators,
            ),
            signals_train_without_bf: signals_train_without_bf,
        }
    }
    pub fn update_bf<'b>(
        &mut self,
        src_transpose: &[Vec<f64>],
        s: &'a SETTINGS,
        fa: &'b FUNCS_EXTRACT_ARGS_TYPE<SETTINGS_SIGNAL, Box<dyn SignalTrain>>,
        indicators_without_bf: &MAP<&'a str, Box<dyn Indicator>>,
    ) {
        self.signals_train = get_signals_from_settings(
            &s.signals_train,
            &s.indications,
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
    pub fn signals_series(
        &self,
        indications: &MAP<&'a str, f64>,
        src_transpose: &[Vec<f64>],
    ) -> MAP<&'a str, f64> {
        unsafe { &*self.settings_signals }
            .iter()
            .fold(MAP::default(), |mut map, setting| {
                let key_uniq_str = setting.0.as_str();
                let mut src_arg = vec![];
                let mut signals_arg = vec![];
                for src_arg_el in &setting.1.used_src {
                    src_arg.push({
                        let sk = &src_transpose[src_arg_el.index];
                        sk[sk.len() - 1 - src_arg_el.sub_from_last_i]
                    });
                }
                for ind_arg_el in &setting.1.used_ind {
                    src_arg.push(indications[ind_arg_el.as_str()]);
                }
                for signals_arg_el in &setting.1.used_signals {
                    signals_arg.push(map[signals_arg_el.as_str()].clone());
                }
                if !setting.1.procedure_used_src.is_empty() {
                    src_arg = setting
                        .1
                        .procedure_used_src
                        .iter()
                        .map(|i| src_arg[*i])
                        .collect();
                }
                if !setting.1.procedure_used_signals.is_empty() {
                    src_arg = setting
                        .1
                        .procedure_used_signals
                        .iter()
                        .map(|i| src_arg[*i])
                        .collect();
                }
                let signal = unsafe { &(&(*self.signals_train).signals_train)[key_uniq_str] };
                map.insert(
                    key_uniq_str,
                    signal
                        .1
                        .signal_with_bf(&src_arg, &signals_arg, &signal.0, 0),
                );
                map
            })
    }
    pub fn signals_vec(
        &self,
        src_transpose: &[Vec<f64>],
    ) -> MAP<&'a str, Vec<f64>> {
        unsafe { &*self.settings_signals }
            .iter()
            .map(|(k, setting)| {
                let key_uniq = k.as_str();
                let signal = unsafe { &(&(*self.signals_train).signals_train)[key_uniq] };
                (
                    key_uniq,
                    signal.1.signals_vec(
                        &get_in_from_settings(
                            &setting.used_ind,
                            &setting.used_src,
                            &setting.procedure_used_src,
                            unsafe { &*self.settings_indicators },
                            src_transpose,
                            unsafe { &(*self.indicators).indicators_without_bf },
                        ),
                        &get_signals_arg_from_settings(
                            &setting.used_signals,
                            &setting.procedure_used_signals,
                            unsafe { &*self.settings_signals },
                            unsafe { &*self.settings_indicators },
                            src_transpose,
                            unsafe { &(*self.signals_train).signals_train_without_bf },
                            unsafe { &(*self.indicators).indicators_without_bf },
                        ),
                    ),
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
    use bc_pack_indicators::FUNCS_EXTRACT_ARGS as FUNCS_EXTRACT_ARGS_IND;
    use bc_pack_signals_train::FUNCS_EXTRACT_ARGS as FUNCS_EXTRACT_ARGS_ST;
    use bc_signals_train::mm::MM;
    use bc_utils_lg::statics::prices::{SRC, SRC_TRANSPOSE};
    use bc_utils_lg::structs::settings::{
        SETTINGS_IND, SETTINGS_INDS, SETTINGS_SIGNAL, SETTINGS_SIGNALS, SETTINGS_USED_USIZE,
    };
    use bc_utils_lg::types::maps::MAP;
    use pretty_assertions::assert_eq as assert_eq_pr;

    use bc_indicators_gw::gw::*;

    #[test]
    fn signals_from_settings_without_bf_res_1() {
        let settings = SETTINGS_SIGNALS::from_iter([(
            "mm_1".to_string(),
            SETTINGS_SIGNAL { key: "mm".to_string(), ..Default::default() },
        )]);
        let funcs_extract_args = FUNCS_EXTRACT_ARGS_ST();
        let res = get_signals_from_settings_without_bf(&settings, &funcs_extract_args);
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
                    used_src: vec![SETTINGS_USED_USIZE { index: 1, sub_from_last_i: 0 }],
                    ..Default::default()
                },
            ),
            (
                "repeat_1".to_string(),
                SETTINGS_IND {
                    key: "repeat".to_string(),
                    kwargs_f64: MAP::from_iter([("value".to_string(), 1.0)]),
                    used_src: vec![SETTINGS_USED_USIZE { index: 1, sub_from_last_i: 0 }],
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
        let indicators = Indicators::new(
            &settings_indicators,
            &FUNCS_EXTRACT_ARGS_IND(),
            &SRC_TRANSPOSE,
        );
        let signals = SignalsTrain::new(
            &settings_signals,
            &settings_indicators,
            &FUNCS_EXTRACT_ARGS_ST(),
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
            df.set_window(10);
            df
        }
        .signal(
            &TREND_MA::default()
                .ind_vec(&SRC)
                .into_iter()
                .map(|v| vec![v])
                .collect::<Vec<Vec<f64>>>(),
            &vec![],
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
                    used_src: vec![SETTINGS_USED_USIZE { index: 1, sub_from_last_i: 0 }],
                    ..Default::default()
                },
            ),
            (
                "repeat_1".to_string(),
                SETTINGS_IND {
                    key: "repeat".to_string(),
                    kwargs_f64: MAP::from_iter([("value".to_string(), 1.0)]),
                    used_src: vec![SETTINGS_USED_USIZE { index: 1, sub_from_last_i: 0 }],
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
        let indicators = Indicators::new(
            &settings_indicators,
            &FUNCS_EXTRACT_ARGS_IND(),
            &SRC_TRANSPOSE,
        );
        let signals = SignalsTrain::new(
            &settings_signals,
            &settings_indicators,
            &FUNCS_EXTRACT_ARGS_ST(),
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
            df.set_window(10);
            df
        }
        .signals_vec(
            &TREND_MA::default()
                .ind_vec(&SRC)
                .into_iter()
                .map(|v| vec![v])
                .collect::<Vec<Vec<f64>>>(),
            &vec![],
        );
        assert_eq_pr!(
            res_1.iter().filter(|v| !v.is_nan()).collect::<Vec<&f64>>(),
            res_2.iter().filter(|v| !v.is_nan()).collect::<Vec<&f64>>()
        );
    }
}
