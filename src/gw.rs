use bc_signals_train::prelude::*;
use bc_utils::other::{procedure_used, transpose, vec_len_sync_set};
use bc_utils_lg::structs::settings::{SETTINGS_INDS, SETTINGS_SIGNAL, SETTINGS_SIGNALS};
use bc_utils_lg::traits::w::{w_scan, w_src, w_sum};
use bc_utils_lg::types::maps::{MAP, MAP_LINK, PACK};

use bc_indicators_gw::gw::Indicators;

pub fn get_src<'a>(
    buffer: &[Vec<f64>],
    indications: &MAP<&str, Vec<f64>>,
    signals_train: &MAP<&str, Vec<f64>>,
    s: &SETTINGS_SIGNAL,
) -> Vec<Vec<f64>> {
    let mut res =
        Vec::with_capacity(s.used_src.len() + s.used_ind.len() + s.used_signals_train.len());
    for used_src in &s.used_src {
        let src = &buffer[used_src.index];
        res.push(src[..src.len() - used_src.sub_from_last_i].to_vec());
    }
    for used_ind in &s.used_ind {
        res.push(indications[used_ind.as_str()].to_vec());
    }
    for used_signals_train in &s.used_signals_train {
        res.push(signals_train[used_signals_train.as_str()].to_vec());
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
    buffer: &[Vec<f64>],
    indications: &MAP<&str, f64>,
    signals_train: &MAP<&str, f64>,
    s: &SETTINGS_SIGNAL,
) -> Vec<f64> {
    let mut res = vec![];
    for src_arg_el in &s.used_src {
        res.push({
            let sk = &buffer[src_arg_el.index];
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

#[derive(Default)]
pub struct SignalsTrain<'a>(pub MAP<&'a str, Box<dyn SignalTrain>>);

impl W for SignalsTrain<'_> {
    fn w(&self) -> usize {
        self.0.values().map(|v| v.w()).max().unwrap()
    }
}

impl<'a> SignalsTrain<'a> {
    pub fn w_map_all(&self, s: &'a SETTINGS_SIGNALS) -> MAP_LINK<&'a str, usize> {
        w_scan(
            self.0.iter(),
            s.iter(),
            |v| v.w(),
            |setting, init, k| {
                [
                    w_src(&setting.used_src),
                    w_sum(&setting.used_signals_train, init),
                    init[k.as_str()],
                ]
            },
        )
    }
    pub fn w_all(&self, s: &SETTINGS_SIGNALS) -> usize {
        self.w_map_all(s).values().max().copied().unwrap()
    }
}

impl<'a> SignalsTrain<'a> {
    pub fn new_empty_bf(
        s: &'a SETTINGS_SIGNALS,
        pack: &PACK<SETTINGS_SIGNAL, Box<dyn SignalTrain>>,
    ) -> Self {
        SignalsTrain(
            s.iter()
                .map(|(signal_name, settings_signal)| {
                    let signal = pack[settings_signal.key.as_str()](settings_signal);
                    (signal_name.as_str(), signal)
                })
                .collect(),
        )
    }
    pub fn init_bf(
        &self,
        buffer: &[Vec<f64>],
        s: &SETTINGS_SIGNALS,
        s_ind: &SETTINGS_INDS,
        indicators: &Indicators,
    ) {
        let indicators = indicators.clone();
        let buffer_vec_trans = transpose(buffer.to_vec());
        let w = buffer_vec_trans.len() - self.w_all(s);
        let (buffer_ind_init, buffer_ind_vec) = (
            transpose(buffer_vec_trans[..w].to_vec()),
            transpose(buffer_vec_trans[w..].to_vec()),
        );
        indicators.init_bf(&buffer_ind_init, s_ind);
        let map_ind = indicators.vec(&buffer_ind_vec, s_ind);
        let mut map_sign = MAP::default();
        for (k, setting) in s.iter() {
            let signal = &self.0[k.as_str()];
            let src = get_src(buffer, &map_ind, &map_sign, setting);
            signal.init_bf(&src[..signal.w()]);
            map_sign.insert(k.as_str(), signal.signals_vec(&src[signal.w()..]));
            signal.init_bf(&src);
        }
    }
    pub fn new(
        buffer: &[Vec<f64>],
        s: &'a SETTINGS_SIGNALS,
        s_ind: &SETTINGS_INDS,
        indicators: &Indicators,
        pack: &PACK<SETTINGS_SIGNAL, Box<dyn SignalTrain>>,
    ) -> Self {
        let bind = SignalsTrain::new_empty_bf(s, pack);
        bind.init_bf(buffer, s, s_ind, indicators);
        bind
    }
}

impl<'a> SignalsTrain<'a> {
    pub fn series(
        &self,
        buffer: &[Vec<f64>],
        s: &'a SETTINGS_SIGNALS,
        indications: &MAP<&str, f64>,
    ) -> MAP<&'a str, f64> {
        s.iter().fold(MAP::default(), |mut init, (k, setting)| {
            let signal = &self.0[k.as_str()];
            init.insert(
                k.as_str(),
                signal.signal(&get_src_series(buffer, indications, &init, setting)),
            );
            init
        })
    }
    pub fn execute_bf(&self) {
        for sign in self.0.values() {
            sign.execute_bf();
        }
    }
    pub fn vec(
        &self,
        buffer: &[Vec<f64>],
        s: &'a SETTINGS_SIGNALS,
        indications: &MAP<&str, Vec<f64>>,
    ) -> MAP<&'a str, Vec<f64>> {
        s.iter().fold(MAP::default(), |mut init, (k, setting)| {
            let signal = &self.0[k.as_str()];
            init.insert(
                k.as_str(),
                signal.signals_vec(&get_src(buffer, indications, &init, setting)),
            );
            init
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;

    use bc_packs::{PACK_IND, PACK_SIGN_TR};
    use bc_signals_train::mm::MM;
    use bc_test_kit::prelude::*;
    use bc_test_kit::settings::signals_train::SIGNALS_TRAIN;

    use bc_utils_lg::types::maps::MAP;
    use pretty_assertions::assert_eq as assert_eq_pr;

    #[test]
    fn new_empty_bf_res_1() {
        let res = SignalsTrain::new_empty_bf(&SIGNALS_TRAIN, &PACK_SIGN_TR);
        let res_1 = res.0.get("mm_1").unwrap().as_ref();
        let mm_test_1 = MM::new(0, 0, 3, 5, 0.0001, 0.01, 0., -1., 1.);
        let mm_test_2 = (res_1 as &dyn Any).downcast_ref::<MM>().unwrap();
        assert_eq_pr!(&mm_test_1, mm_test_2);
    }

    #[test]
    fn w_all_res_1() {
        assert_eq_pr!(
            SignalsTrain::new_empty_bf(&SIGNALS_TRAIN, &PACK_SIGN_TR).w_all(&SIGNALS_TRAIN,),
            6
        );
    }

    #[test]
    fn get_src_res_1() {
        let indicators = Indicators::new_empty_bf(&INDICATIONS, &PACK_IND);
        let w_all = indicators.w_all(&INDICATIONS);
        indicators.init_bf(&transpose(SRC[..w_all].to_vec()), &INDICATIONS);
        let indications = indicators.vec(&transpose(SRC[w_all..].to_vec()), &INDICATIONS);
        assert_eq_pr!(
            get_src(
                &SRC_TRANSPOSE,
                &indications,
                &Default::default(),
                &SIGNALS_TRAIN["mm_1"]
            ),
            transpose(vec![
                OPEN[indications["rma_1"].len() - SIGNALS_TRAIN["mm_1"].used_src[0].sub_from_last_i
                    ..OPEN.len() - SIGNALS_TRAIN["mm_1"].used_src[0].sub_from_last_i]
                    .to_vec(),
                indications["rma_1"].to_vec()
            ])
        );
    }

    #[test]
    fn get_src_series_res_1() {
        assert_eq_pr!(
            get_src_series(
                &SRC_TRANSPOSE,
                &MAP::from_iter([("rma_1", 1.)]),
                &Default::default(),
                &SIGNALS_TRAIN["mm_1"]
            ),
            vec![SRC_EL1[1], 1.,]
        )
    }

    #[test]
    fn init_bf_res_1() {
        let signals_train = SignalsTrain::new_empty_bf(&SIGNALS_TRAIN, &PACK_SIGN_TR);
        let indicators = Indicators::new_empty_bf(&INDICATIONS, &PACK_IND);
        let buffer_vec_trans = SRC[..49].to_vec();
        let w = buffer_vec_trans.len() - signals_train.w_all(&SIGNALS_TRAIN);
        let (buffer_ind_init, buffer_ind_vec) = (
            transpose(buffer_vec_trans[..w].to_vec()),
            transpose(buffer_vec_trans[w..].to_vec()),
        );
        indicators.init_bf(&buffer_ind_init, &INDICATIONS);
        let map_ind = indicators.vec(&buffer_ind_vec, &INDICATIONS);
        signals_train.init_bf(
            &transpose(buffer_vec_trans),
            &SIGNALS_TRAIN,
            &INDICATIONS,
            &indicators,
        );
        let res = signals_train.0["mm_1"].clone();
        res.init_bf(&get_src(
            &buffer_ind_vec,
            &map_ind,
            &Default::default(),
            &SIGNALS_TRAIN["mm_1"],
        ));
        assert_eq_pr!(
            signals_train.series(
                &SRC_TRANSPOSE,
                &SIGNALS_TRAIN,
                &indicators.series(&SRC_TRANSPOSE, &INDICATIONS)
            )["mm_1"],
            res.signal(&[OPEN_LAST])
        );
    }

    #[test]
    fn series_res_1() {
        let signals_train = SignalsTrain::new_empty_bf(&SIGNALS_TRAIN, &PACK_SIGN_TR);
        let indicators = Indicators::new_empty_bf(&INDICATIONS, &PACK_IND);
        indicators.init_bf(&SRC_TRANSPOSE, &INDICATIONS);
        signals_train.init_bf(&SRC_TRANSPOSE, &SIGNALS_TRAIN, &INDICATIONS, &indicators);
        let indications = indicators.series(&SRC_TRANSPOSE, &INDICATIONS);
        assert_eq_pr!(
            signals_train.series(&SRC_TRANSPOSE, &SIGNALS_TRAIN, &indications)["mm_1"],
            signals_train.0["mm_1"].signal(&get_src_series(
                &SRC_TRANSPOSE,
                &indications,
                &Default::default(),
                &SIGNALS_TRAIN["mm_1"]
            ))
        );
    }

    #[test]
    fn vec_res_1() {
        let signals_train = SignalsTrain::new_empty_bf(&SIGNALS_TRAIN, &PACK_SIGN_TR);
        let indicators = Indicators::new_empty_bf(&INDICATIONS, &PACK_IND);
        indicators.init_bf(&SRC_TRANSPOSE, &INDICATIONS);
        signals_train.init_bf(&SRC_TRANSPOSE, &SIGNALS_TRAIN, &INDICATIONS, &indicators);
        let indications = indicators.vec(&SRC_TRANSPOSE, &INDICATIONS);
        assert_eq_pr!(
            signals_train.vec(&SRC_TRANSPOSE, &SIGNALS_TRAIN, &indications)["mm_1"],
            signals_train.0["mm_1"].signals_vec(&get_src(
                &SRC_TRANSPOSE,
                &indications,
                &Default::default(),
                &SIGNALS_TRAIN["mm_1"]
            ))
        );
    }
}
