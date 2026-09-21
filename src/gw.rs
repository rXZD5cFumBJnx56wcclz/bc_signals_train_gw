use bc_gw_utils::prelude::*;
use bc_signals_train::prelude::*;
use bc_utils::other::transpose;
use bc_utils_lg::prelude::*;

use bc_indicators_gw::gw::Indicators;

#[derive(Default, Clone)]
pub struct SignalsTrain<'a>(pub MAP<&'a str, Box<dyn SignalTrain>>);

impl W for SignalsTrain<'_> {
    fn w(&self) -> usize {
        self.0.values().map(|v| v.w()).max().unwrap_or_default()
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
        self.w_map_all(s)
            .values()
            .max()
            .copied()
            .unwrap_or_default()
    }
}

impl<'a> SignalsTrain<'a> {
    pub fn init_empty(
        &mut self,
        s: &'a SETTINGS_SIGNALS,
        pack: &PACK<SETTINGS_SIGNAL, Box<dyn SignalTrain>>,
    ) {
        *self = SignalsTrain(
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
        if indicators.w() != 0 {
            indicators.init_bf(&buffer_ind_init, s_ind);
        }
        let map_ind = indicators.vec(&buffer_ind_vec, s_ind);
        let mut map_sign = MAP::default();
        for (k, setting) in s.iter() {
            let signal = &self.0[k.as_str()];
            let mut src = SrcGw::default();
            src.push_vec(&buffer, &setting.used_src);
            src.push_map(&map_ind, &setting.used_ind);
            src.push_map(&map_sign, &setting.used_signals_train);
            src.all_check(&setting.procedure_used_src);
            signal.init_bf(&src[..signal.w()]);
            map_sign.insert(k.as_str(), signal.signals_vec(&src[signal.w()..]));
            signal.init_bf(&src);
        }
    }
    pub fn init(
        &mut self,
        buffer: &[Vec<f64>],
        s: &'a SETTINGS_SIGNALS,
        s_ind: &SETTINGS_INDS,
        indicators: &Indicators,
        pack: &PACK<SETTINGS_SIGNAL, Box<dyn SignalTrain>>,
    ) {
        self.init_empty(s, pack);
        self.init_bf(buffer, s, s_ind, indicators);
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
            let mut src = SrcGwSeries::default();
            src.push_vec(buffer, &setting.used_src);
            src.push_map(indications, &setting.used_ind);
            src.push_map(&init, &setting.used_signals_train);
            src.all_check(&setting.procedure_used_src);
            init.insert(k.as_str(), signal.signal(&src));
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
            let mut src = SrcGw::default();
            src.push_vec(buffer, &setting.used_src);
            src.push_map(indications, &setting.used_ind);
            src.push_map(&init, &setting.used_signals_train);
            src.all_check(&setting.procedure_used_src);
            init.insert(k.as_str(), signal.signals_vec(&src));
            init
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use bc_packs::{PACK_IND, PACK_SIGN_TR};
    use bc_signals_train::mm::MM;
    use bc_test_kit::prelude::*;
    use bc_utils_lg::test_state::prelude::*;

    #[test]
    fn new_empty_bf_res_1() {
        let mut res = SignalsTrain::default();
        res.init_empty(&SIGNALS_TRAIN, &PACK_SIGN_TR);
        let res_1 = res.0.get("mm_1").unwrap().as_ref();
        let mm_test_1 = MM::new(0, 0, 3, 5, 0.0001, 0.01, 0., -1., 1.);
        let mm_test_2 = (res_1 as &dyn Any).downcast_ref::<MM>().unwrap();
        assert_eq_pr!(&mm_test_1, mm_test_2);
    }

    #[test]
    fn w_all_res_1() {
        let mut res = SignalsTrain::default();
        res.init_empty(&SIGNALS_TRAIN, &PACK_SIGN_TR);
        assert_eq_pr!(res.w_all(&SIGNALS_TRAIN,), 6);
    }

    #[test]
    fn init_bf_res_1() {
        let mut indicators = Indicators::default();
        indicators.init_empty(&INDICATIONS, &PACK_IND);
        let mut signals_train = SignalsTrain::default();
        signals_train.init_empty(&SIGNALS_TRAIN, &PACK_SIGN_TR);
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
        let mut src_test = SrcGw::default();
        src_test.push_vec(&buffer_ind_vec, &SIGNALS_TRAIN["mm_1"].used_src);
        src_test.push_map(&map_ind, &SIGNALS_TRAIN["mm_1"].used_ind);
        src_test.all_check(&SIGNALS_TRAIN["mm_1"].procedure_used_src);
        res.init_bf(&src_test);
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
        let mut indicators = Indicators::default();
        indicators.init_empty(&INDICATIONS, &PACK_IND);
        let mut signals_train = SignalsTrain::default();
        signals_train.init_empty(&SIGNALS_TRAIN, &PACK_SIGN_TR);
        indicators.init_bf(&SRC_TRANSPOSE, &INDICATIONS);
        signals_train.init_bf(&SRC_TRANSPOSE, &SIGNALS_TRAIN, &INDICATIONS, &indicators);
        let indications = indicators.series(&SRC_TRANSPOSE, &INDICATIONS);
        let mut src_test = SrcGwSeries::default();
        src_test.push_vec(&SRC_TRANSPOSE, &SIGNALS_TRAIN["mm_1"].used_src);
        src_test.push_map(&indications, &SIGNALS_TRAIN["mm_1"].used_ind);
        src_test.all_check(&SIGNALS_TRAIN["mm_1"].procedure_used_src);
        assert_eq_pr!(
            signals_train.series(&SRC_TRANSPOSE, &SIGNALS_TRAIN, &indications)["mm_1"],
            signals_train.0["mm_1"].signal(&src_test)
        );
    }

    #[test]
    fn vec_res_1() {
        let mut indicators = Indicators::default();
        indicators.init_empty(&INDICATIONS, &PACK_IND);
        let mut signals_train = SignalsTrain::default();
        signals_train.init_empty(&SIGNALS_TRAIN, &PACK_SIGN_TR);
        indicators.init_bf(&SRC_TRANSPOSE, &INDICATIONS);
        signals_train.init_bf(&SRC_TRANSPOSE, &SIGNALS_TRAIN, &INDICATIONS, &indicators);
        let indications = indicators.vec(&SRC_TRANSPOSE, &INDICATIONS);
        let mut src_test = SrcGw::default();
        src_test.push_vec(&SRC_TRANSPOSE, &SIGNALS_TRAIN["mm_1"].used_src);
        src_test.push_map(&indications, &SIGNALS_TRAIN["mm_1"].used_ind);
        src_test.all_check(&SIGNALS_TRAIN["mm_1"].procedure_used_src);
        assert_eq_pr!(
            signals_train.vec(&SRC_TRANSPOSE, &SIGNALS_TRAIN, &indications)["mm_1"],
            signals_train.0["mm_1"].signals_vec(&src_test)
        );
    }
}
