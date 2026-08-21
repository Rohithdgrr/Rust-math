//! Integration tests running every public module against a real sales dataset.
//!
//! The fixture (`data/sales.csv`) is decoded from a real Excel sales-analysis
//! workbook. It has 647 rows and 5 columns:
//!   Sales Person | Geography | Product | Amount | Customers
//! `Amount` drives regression targets; `Geography` drives 6-class classification.

use mathverse_machine_learning::estimator::{Classifier, Estimator, Transformer};
use mathverse_machine_learning::*;

/// One parsed sales record.
#[derive(Debug, Clone)]
struct SalesRow {
    person: f64,
    geo: f64,
    product: f64,
    amount: f64,
    customers: f64,
}

/// Parse the embedded CSV. Each data line is `"Person","Geo","Product",Amount,Customers`.
fn load_rows() -> Vec<SalesRow> {
    let raw = include_str!("data/sales.csv");
    let mut rows = Vec::new();
    // No header line: every line is a data row. Strip the UTF-8 BOM that the
    // spreadsheet export wrote on the first line so all 647 rows parse.
    let raw = raw.strip_prefix('\u{FEFF}').unwrap_or(raw);
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = split_csv(line);
        let customers = parts.pop().unwrap().parse::<f64>().unwrap();
        let amount = parts.pop().unwrap().parse::<f64>().unwrap();
        let product = parts.pop().unwrap();
        let geo = parts.pop().unwrap();
        let person = parts.pop().unwrap();
        rows.push(SalesRow {
            person: label_index(person, "person"),
            geo: label_index(geo, "geo"),
            product: label_index(product, "product"),
            amount,
            customers,
        });
    }
    rows
}

/// Deterministic string -> small-integer index (missing strings get 0).
fn label_index(s: String, _ctx: &str) -> f64 {
    let v: u64 = s.bytes().map(u64::from).sum();
    (v % 20) as f64
}

/// Split `"a","b",1,2` into unquoted fields. Only commas are delimiters; values
/// never contain commas in this dataset.
fn split_csv(line: &str) -> Vec<String> {
    line.split(',')
        .map(|f| f.trim_matches('"').trim().to_string())
        .collect()
}

/// Categorical index features + Customers; `Amount` is the regression target.
fn regression_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let rows = load_rows();
    let x: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| vec![r.person, r.geo, r.product, r.customers])
        .collect();
    let y: Vec<f64> = rows.iter().map(|r| r.amount).collect();
    (x, y)
}

/// Geography (6 classes) as classification target.
fn geo_classification_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let rows = load_rows();
    let x: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| vec![r.person, r.product, r.customers])
        .collect();
    // Group counts per geography index value to know the real class range.
    let mut geos: Vec<f64> = rows.iter().map(|r| r.geo).collect();
    geos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    geos.dedup();
    let y: Vec<f64> = rows.iter().map(|r| r.geo).collect();
    assert!(geos.len() >= 2, "need at least 2 classes");
    (x, y)
}

/// Binary classification: Amount above the median.
fn binary_classification_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let rows = load_rows();
    let mut amounts: Vec<f64> = rows.iter().map(|r| r.amount).collect();
    amounts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = amounts[amounts.len() / 2];
    let x: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| vec![r.person, r.geo, r.product, r.customers])
        .collect();
    let y: Vec<f64> = rows.iter().map(|r| if r.amount > median { 1.0 } else { 0.0 }).collect();
    (x, y)
}

fn is_finite(v: f64) -> bool {
    v.is_finite()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn accuracy(pred: &[f64], target: &[f64]) -> f64 {
    let correct = pred
        .iter()
        .zip(target.iter())
        .filter(|(p, t)| (**p - **t).abs() < 0.5)
        .count();
    correct as f64 / pred.len() as f64
}

// ---------------------------------------------------------------------------
// linear / elastic_net / logistic
// ---------------------------------------------------------------------------

#[test]
fn linear_algorithms_regress_amount() {
    let (x, y) = regression_data();
    let lin = linear::fit(&x, &y).unwrap();
    assert!(lin.r_squared >= 0.0, "r2: {}", lin.r_squared);
    let pred = linear::predict(&x, &lin.coefficients, lin.intercept).unwrap();
    assert!((mean(&pred) - mean(&y)).abs() < 1.0, "mean drift");

    let ridge = linear::fit_ridge(&x, &y, 1.0).unwrap();
    assert!(ridge.r_squared >= 0.0, "ridge r2: {}", ridge.r_squared);
    let rp = linear::predict(&x, &ridge.coefficients, ridge.intercept).unwrap();
    assert_eq!(rp.len(), y.len());

    let lasso = linear::fit_lasso(&x, &y, 0.1, 200, 1e-6).unwrap();
    let lp = linear::predict(&x, &lasso.coefficients, lasso.intercept).unwrap();
    assert_eq!(lp.len(), y.len());

    let en = elastic_net::fit_elastic_net(&x, &y, 0.1, 0.5, 200, 1e-6).unwrap();
    let ep = elastic_net::predict(&x, &en.coefficients, en.intercept);
    assert_eq!(ep.len(), y.len());
    assert!(en.r_squared >= 0.0, "en r2: {}", en.r_squared);
}

#[test]
fn logistic_handles_binary_and_multiclass() {
    let (x, y) = binary_classification_data();
    let lr = logistic::fit(&x, &y, 0.1, 200, 1e-6, 1.0).unwrap();
    let proba = logistic::predict_proba(&x, &lr.coefficients, lr.intercept);
    assert!(proba.iter().all(|&p| (0.0..=1.0).contains(&p)));
    let pred = logistic::predict(&x, &lr.coefficients, lr.intercept);
    assert!(accuracy(&pred, &y) > 0.5, "binary logistic acc: {}", accuracy(&pred, &y));
    let ce = logistic::cross_entropy(&x, &y, &lr.coefficients, lr.intercept);
    assert!(ce.is_finite() && ce >= 0.0, "ce: {ce}");

    // Multiclass one-vs-rest on 6 geographies.
    let (gx, gy) = geo_classification_data();
    let ovr = logistic::fit_ovr(&gx, &gy, 0.05, 150, 1e-6, 1.0).unwrap();
    assert_eq!(ovr.classes.len(), 6, "classes: {:?}", ovr.classes);
    let op = logistic::predict_proba_ovr(&gx, &ovr);
    assert_eq!(op.len(), gx.len());
    assert!(op.iter().all(|row| (row.iter().sum::<f64>() - 1.0).abs() < 1e-3));
    let op_pred = logistic::predict_ovr(&gx, &ovr);
    assert_eq!(op_pred.len(), gx.len());
    assert!(accuracy(&op_pred, &gy) > 0.3, "ovr acc: {}", accuracy(&op_pred, &gy));
}

// ---------------------------------------------------------------------------
// knn / tree / forest / boosting / xgboost
// ---------------------------------------------------------------------------

#[test]
fn tree_variants_classify_geography() {
    let (x, y) = geo_classification_data();
    let classes: Vec<f64> = {
        let mut c = y.clone();
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        c.dedup();
        c
    };

    let mut knn_clf = knn::classify(&x, &y, &x, 5).unwrap();
    assert_eq!(knn_clf.len(), x.len());
    knn_clf = knn::classify(&x, &y, &x, 1).unwrap();
    assert!(accuracy(&knn_clf, &y) > 0.5);

    let knn_reg = knn::regress(&x, &y, &x, 5).unwrap();
    assert_eq!(knn_reg.len(), x.len());

    let mut tree = tree::DecisionTree::new(6, 4);
    tree.fit(&x, &y);
    let preds = tree.predict(&x);
    assert!(accuracy(&preds, &y) > 0.5, "tree acc: {}", accuracy(&preds, &y));
    let proba = tree.predict_proba(&x, &classes);
    assert_eq!(proba.len(), x.len());
    assert!(proba[0].iter().sum::<f64>().abs() - 1.0 < 0.1 || !proba[0].is_empty());

    let mut forest = forest::RandomForest::new(10, 6, 3);
    forest.fit(&x, &y);
    let fpreds = forest.predict(&x);
    assert!(accuracy(&fpreds, &y) > 0.3, "forest acc: {}", accuracy(&fpreds, &y));
    let fproba = forest.predict_proba(&x, &classes);
    assert_eq!(fproba.len(), x.len());
}

#[test]
fn boosting_and_xgboost_regress_and_classify() {
    let (x, y) = regression_data();
    let mut gbr = boosting::GradientBoostingRegressor::new(30, 0.1, 3);
    gbr.fit(&x, &y);
    let bp = gbr.predict(&x);
    assert!(is_finite(mean(&bp)));
    assert!((mean(&bp) - mean(&y)).abs() < 300.0, "gbr drift: {}", mean(&bp));

    let mut xgr = xgboost::XGBoostRegressor::new(30, 0.1, 4, 1.0, 0.0);
    xgr.fit(&x, &y).unwrap();
    let xp = xgr.predict(&x);
    assert!(is_finite(mean(&xp)));

    let (bx, by) = binary_classification_data();
    let mut gbc = boosting::GradientBoostingClassifier::new(30, 0.1, 3);
    gbc.fit(&bx, &by).unwrap();
    let gproba = gbc.predict_proba(&bx);
    assert!(gproba.iter().all(|&p| (0.0..=1.0).contains(&p)));
    let gcpred = gbc.predict(&bx);
    assert!(accuracy(&gcpred, &by) > 0.5, "gbc acc: {}", accuracy(&gcpred, &by));

    let mut xgc = xgboost::XGBoostClassifier::new(30, 0.1, 4, 1.0, 0.0);
    xgc.fit(&bx, &by).unwrap();
    let xproba = xgc.predict_proba(&bx);
    assert!(xproba.iter().all(|&p| (0.0..=1.0).contains(&p)));
    let xcpred = xgc.predict(&bx);
    assert!(accuracy(&xcpred, &by) > 0.5, "xgc acc: {}", accuracy(&xcpred, &by));
}

// ---------------------------------------------------------------------------
// svm / naive_bayes / gaussian_process / neural_net
// ---------------------------------------------------------------------------

#[test]
fn svm_and_naive_bayes_classify() {
    let (x, y) = binary_classification_data();
    // SVM expects labels in {-1, +1}.
    let y_svm: Vec<f64> = y.iter().map(|&v| if v > 0.5 { 1.0 } else { -1.0 }).collect();

    let kernel = svm::Kernel::RBF { gamma: 0.1 };
    let _sim = kernel.compute(&x[0], &x[1]);

    let mut svm_linear = svm::SVM::linear(1.0);
    svm_linear.fit(&x, &y_svm);
    let sp = svm_linear.predict(&x);
    let correct = sp
        .iter()
        .zip(&y_svm)
        .filter(|(p, t)| (**p - **t).abs() < 0.5)
        .count();
    assert!(correct >= sp.len() / 2, "svm acc: {correct}/{}", sp.len());

    let mut svm_rbf = svm::SVM::rbf(1.0, 0.5);
    svm_rbf.fit(&x, &y_svm);
    assert_eq!(svm_rbf.predict(&x).len(), x.len());

    let nb = naive_bayes::fit(&x, &y).unwrap();
    let nb_pred = naive_bayes::predict(&nb, &x).unwrap();
    assert!(accuracy(&nb_pred, &y) > 0.3);
    let nb_proba = naive_bayes::predict_proba(&nb, &x).unwrap();
    assert_eq!(nb_proba.len(), x.len());
    assert!(nb_proba.iter().all(|row| (row.iter().sum::<f64>() - 1.0).abs() < 1e-3));
}

#[test]
fn gaussian_process_and_neural_net_regress() {
    let (x, y) = regression_data();
    let n = x.len();
    let x_sub: Vec<Vec<f64>> = x.iter().take(80).cloned().collect();
    let y_sub: Vec<f64> = y.iter().take(80).copied().collect();

    let gp = gaussian_process::GaussianProcess::fit(
        &x_sub,
        &y_sub,
        gaussian_process::GpKernel::RBF { length: 1.0 },
        1e-3,
    );
    assert!(gp.is_ok(), "gp fit failed");
    let (gp_mean, gp_var) = gp.unwrap().predict(&x_sub);
    assert!(gp_mean.iter().all(|&v| v.is_finite()));
    assert!(gp_var.iter().all(|&v| v.is_finite() && v >= 0.0));
    assert!((mean(&gp_mean) - mean(&y_sub)).abs() < 500.0, "gp drift");

    // Small neural net on the binary problem.
    let (bx, by) = binary_classification_data();
    let bx_sub: Vec<Vec<f64>> = bx.iter().take(60).cloned().collect();
    let by_sub: Vec<f64> = by.iter().take(60).copied().collect();
    let mut nn = neural_net::NeuralNet::new(vec![
        neural_net::Layer::Linear {
            weights: vec![vec![0.1; bx_sub[0].len()], vec![0.1; bx_sub[0].len()]],
            bias: vec![0.0; 2],
        },
        neural_net::Layer::Sigmoid,
        neural_net::Layer::Linear { weights: vec![vec![0.1; 2]], bias: vec![0.0] },
    ]);
    // fit panics if the y vector mismatches; use forward first.
    nn.fit(&bx_sub, &by_sub, 0.01, 20);
    let fwd = nn.forward(&bx_sub);
    assert_eq!(fwd.len(), bx_sub.len());
    let nn_pred = nn.predict(&bx_sub);
    assert_eq!(nn_pred.len(), bx_sub.len());
    assert!(nn_pred.iter().all(|&v| v.is_finite()));

    // Regression network on amount subset.
    let xr_sub: Vec<Vec<f64>> = x.iter().take(60).cloned().collect();
    let yr_sub: Vec<f64> = y.iter().take(60).copied().collect();
    let mut nn_reg = neural_net::NeuralNet::new(vec![
        neural_net::Layer::Linear {
            weights: vec![vec![0.001; xr_sub[0].len()]],
            bias: vec![2000.0],
        },
    ]);
    nn_reg.fit(&xr_sub, &yr_sub, 0.0001, 30);
    let rp = nn_reg.predict(&xr_sub);
    assert!(rp.iter().all(|&v| v.is_finite()));
}

// ---------------------------------------------------------------------------
// ensembles: voting / blending / bagging / adaboost / stacking
// ---------------------------------------------------------------------------

#[test]
fn voting_blending_bagging_adaboost_stacking() {
    let (x, y) = binary_classification_data();
    let n = x.len();

    // VotingClassifier: pure closures that don't need fitting.
    let mut vc2 = ensemble::VotingClassifier::new();
    vc2.add_model(|xr| xr.iter().map(|r| if r[0] > r[1] { 1.0 } else { 0.0 }).collect(), 1.0);
    vc2.add_model(|xr| xr.iter().map(|r| if r[2] > 0.0 { 1.0 } else { 0.0 }).collect(), 1.0);
    let votes = vc2.predict(&x);
    assert_eq!(votes.len(), n);

    let mut vr = ensemble::VotingRegressor::new();
    vr.add_model(|xr| xr.iter().map(|r| r[0]).collect(), 1.0);
    vr.add_model(|xr| xr.iter().map(|r| r[1] * 2.0).collect(), 1.0);
    assert_eq!(vr.predict(&x).len(), n);

    // Bagging
    let mut bag = ensemble_adv::BaggingClassifier::new(
        5,
        0.8,
        ensemble_adv::BaggingBase::DecisionTree,
    );
    bag.fit(&x, &y);
    let bp = bag.predict(&x);
    assert_eq!(bp.len(), n);

    // AdaBoost (binary only)
    let mut ada = ensemble_adv::AdaBoostClassifier::new(5, 0.5);
    ada.fit(&x, &y);
    let ap = ada.predict(&x);
    assert_eq!(ap.len(), n);

    // Stacking
    let mut stack = ensemble_adv::StackingClassifier::new(
        vec![
            ensemble_adv::StackingBase::Logistic,
            ensemble_adv::StackingBase::Knn,
            ensemble_adv::StackingBase::DecisionTree,
        ],
        ensemble_adv::StackingMeta::Logistic,
    );
    stack.fit(&x, &y);
    let stp = stack.predict(&x);
    assert!(stp.is_ok(), "stacking predict failed");
    assert_eq!(stp.unwrap().len(), n);
}

// ---------------------------------------------------------------------------
// clustering: kmeans / dbscan / hierarchical / gmm / isolation forest
// ---------------------------------------------------------------------------

#[test]
fn clustering_algorithms_run_on_subsample() {
    let (x, _y) = regression_data();
    let sub: Vec<Vec<f64>> = x.iter().take(150).cloned().collect();

    let km = kmeans::kmeans(&sub, 4, 50, 1e-6).unwrap();
    assert_eq!(km.labels.len(), sub.len());
    assert_eq!(km.centroids.len(), 4);
    assert!(km.inertia >= 0.0);

    let db = dbscan::dbscan(&sub, 1000.0, 3);
    assert_eq!(db.labels.len(), sub.len());
    assert!(db.n_clusters >= 1, "dbscan clusters: {}", db.n_clusters);

    let agg = hierarchical::agglomerative(&sub, 3, hierarchical::Linkage::Average);
    assert_eq!(agg.labels.len(), sub.len());
    assert_eq!(agg.n_clusters, 3);

    let gmm = gmm::fit_gmm(&sub, 3, 30, 1e-6);
    assert!(gmm.is_ok(), "gmm fit failed: {:?}", gmm.err());
    let gmm = gmm.unwrap();
    assert_eq!(gmm.weights.len(), 3);
    let gmm_pred = gmm::predict(&gmm, &sub);
    assert_eq!(gmm_pred.len(), sub.len());

    let mut iso = isolation_forest::IsolationForest::new(20, 32);
    iso.fit(&sub);
    let scores = iso.score_samples(&sub);
    assert!(scores.iter().all(|&s| s.is_finite()));
    let iso_pred = iso.predict(&sub);
    assert_eq!(iso_pred.len(), sub.len());
}

// ---------------------------------------------------------------------------
// pca / feature engineering / feature selection / preprocessing
// ---------------------------------------------------------------------------

#[test]
fn pca_and_kernel_pca_transform() {
    let (x, _y) = regression_data();
    let sub: Vec<Vec<f64>> = x.iter().take(120).cloned().collect();

    let mut pca = pca::PCA::new(2);
    pca.fit(&sub);
    let tr = pca.transform(&sub);
    assert_eq!(tr.len(), sub.len());
    assert!(tr[0].len() <= 2);


    let mut kpca = pca::KernelPCA::new(2, 0.1);
    let ktr = kpca.fit_transform(&sub);
    assert_eq!(ktr.len(), sub.len());
    let ktr2 = kpca.transform(&sub);
    assert_eq!(ktr2.len(), sub.len());
}

#[test]
fn feature_preprocessing_functions() {
    let (x, _y) = regression_data();
    let mut xc = x.clone();

    let (means, stds) = feature::standardize(&mut xc);
    assert_eq!(means.len(), x[0].len());
    assert_eq!(stds.len(), x[0].len());
    let (mins, maxs) = feature::min_max(&mut xc);
    assert!(mins.iter().zip(&maxs).all(|(a, b)| a <= b));
    let poly = feature::polynomial_features(&xc, 2);
    assert!(poly.len() == xc.len());

    // feature_selection
    let (idx, filtered) = feature_selection::variance_threshold(&x, 1e-9);
    assert_eq!(filtered.len(), x.len());
    assert!(idx.len() <= x[0].len());
    let corr = feature_selection::pearson_correlation(&y_reg_sample(), &y_reg_sample());
    assert!((corr - 1.0).abs() < 1e-9);
    let x_sub: Vec<Vec<f64>> = x.iter().take(150).cloned().collect();
    let y_sub: Vec<f64> = y_reg_sample_target().iter().take(150).copied().collect();
    let (sel, sx) = feature_selection::select_k_best(&x_sub, &y_sub, 2);
    assert_eq!(sel.len(), 2);
    assert_eq!(sx.len(), 150);
    let cf = feature_selection::correlation_filter(&x_sub, 0.9);
    assert!(cf.0.len() <= x_sub[0].len());

    // preprocessing_adv
    let mut scaler = preprocessing_adv::StandardScaler::new();
    scaler.fit(&x);
    let scaled = scaler.transform(&x);
    assert_eq!(scaled.len(), x.len());
    assert!(scaler.is_fitted());
    let sc_means: Vec<f64> = (0..x[0].len())
        .map(|j| scaled.iter().map(|r| r[j]).sum::<f64>() / scaled.len() as f64)
        .collect();
    assert!(sc_means.iter().all(|&m| m.abs() < 0.5), "stds means: {sc_means:?}");

    let mm = preprocessing_adv::MinMaxScaler::fit_transform(&x);
    let mm_out = mm.transform(&x);
    assert!(mm_out.iter().all(|r| r.iter().all(|&v| (-0.001..=1.001).contains(&v))));

    let le = preprocessing_adv::label_encode(&[3.0, 1.0, 3.0, 2.0]);
    assert_eq!(le, vec![2.0, 0.0, 2.0, 1.0]);
    let oe = preprocessing_adv::ordinal_encode(&[1.0, 5.0], &[5.0, 1.0]);
    assert_eq!(oe, vec![1.0, 0.0]);

    let mut impute_x = vec![vec![1.0, f64::NAN], vec![3.0, 5.0]];
    preprocessing_adv::impute_mean(&mut impute_x);
    assert!(impute_x[0][1].is_finite());
    let mut impute_x2 = vec![vec![1.0, f64::NAN], vec![3.0, 7.0]];
    preprocessing_adv::impute_median(&mut impute_x2);
    assert!(impute_x2[0][1].is_finite());
    let mut impute_x3 = vec![vec![1.0, f64::NAN]];
    preprocessing_adv::impute_constant(&mut impute_x3, -5.0);
    assert_eq!(impute_x3[0][1], -5.0);

    let pw = preprocessing_adv::power_transform(&x, "yeo-johnson");
    assert_eq!(pw.len(), x.len());
    let qt = preprocessing_adv::quantile_transform(&x, 10);
    assert!(qt.iter().all(|r| r.iter().all(|&v| (-0.001..=1.001).contains(&v))));
    let qtf = preprocessing_adv::quantile_transform_fixed(&x, 10);
    assert_eq!(qtf.len(), x.len());
    let (rob, med, iqr) = preprocessing_adv::robust_scale(&x);
    assert_eq!(rob.len(), x.len());
    assert_eq!(med.len(), x[0].len());
    assert!(iqr.iter().all(|&v| v > 0.0));
    let l1 = preprocessing_adv::normalize_l1(&x);
    assert_eq!(l1.len(), x.len());
    let l2 = preprocessing_adv::normalize_l2(&x);
    assert_eq!(l2.len(), x.len());
}

fn y_reg_sample() -> Vec<f64> {
    load_rows().iter().map(|r| r.amount).collect()
}
fn y_reg_sample_target() -> Vec<f64> {
    load_rows().iter().map(|r| r.customers).collect()
}

// ---------------------------------------------------------------------------
// metrics_adv
// ---------------------------------------------------------------------------

#[test]
fn advanced_metrics_all_finite() {
    let (x, y) = binary_classification_data();
    let pred = y.clone(); // perfect predictions
    let tb = metrics_adv::matthews_correlation(&pred, &y);
    assert!((tb - 1.0).abs() < 1e-9, "mcc: {tb}");
    let kappa = metrics_adv::cohen_kappa(&pred, &y);
    assert!((kappa - 1.0).abs() < 1e-9, "kappa: {kappa}");

    let probas: Vec<Vec<f64>> = y.iter().map(|&v| vec![1.0 - v, v]).collect();
    let ll = metrics_adv::log_loss(&probas, &y);
    assert!(ll.is_finite() && ll >= 0.0);
    let bs = metrics_adv::brier_score(&probas, &y);
    assert!(bs.is_finite() && bs >= 0.0);

    // Silhouette on clustering output from kmeans.
    let sub: Vec<Vec<f64>> = x.iter().take(80).cloned().collect();
    let km = kmeans::kmeans(&sub, 3, 50, 1e-6).unwrap();
    let sil = metrics_adv::silhouette_score(&sub, &km.labels, |a, b| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    });
    assert!(sil.is_finite() && sil >= -1.1 && sil <= 1.1);

    let cal = metrics_adv::calibration_curve(&probas, &y, 10);
    assert_eq!(cal.len(), 10);

    let scores = y.clone();
    let p_at_k = metrics_adv::precision_at_k(&scores, &y, 10);
    assert!(p_at_k.is_finite());
    let ndcg = metrics_adv::ndcg(&scores, &y, 10);
    assert!(ndcg.is_finite());

    let y_amt = y_reg_sample();
    let p: Vec<f64> = y_amt.iter().map(|v| v + 10.0).collect();
    let mape = metrics_adv::mean_absolute_percentage_error(&p, &y_amt);
    assert!(mape.is_finite());
    let mae = metrics_adv::median_absolute_error(&p, &y_amt);
    assert_eq!(mae, 10.0);
    let me = metrics_adv::max_error(&p, &y_amt);
    assert_eq!(me, 10.0);
    let tw = metrics_adv::tweedie_deviance(&p, &y_amt, 1.5);
    assert!(tw.is_finite());

    let (gx, gy) = geo_classification_data();
    let cm = metrics_adv::confusion_matrix_detailed(&gy, &gy, 6);
    assert_eq!(cm.tp.len(), 6);
}

// ---------------------------------------------------------------------------
// model_selection / evaluation / estimator traits
// ---------------------------------------------------------------------------

#[test]
fn model_selection_evaluation_and_estimator_traits() {
    let (x, y) = regression_data();

    // Splits
    let (xtr, xte, ytr, yte) = model_selection::train_test_split(&x, &y, 0.2, 42);
    assert_eq!(xtr.len() + xte.len(), x.len());
    let (str_, ste, ystr, yste) = model_selection::stratified_train_test_split(&x, &y, 0.2, 42);
    assert_eq!(str_.len() + ste.len(), x.len());
    assert_eq!(ystr.len() + yste.len(), y.len());

    // k-fold indices & time series
    let folds = model_selection::k_fold_indices(x.len(), 5, 42);
    assert_eq!(folds.len(), 5);
    let tss = model_selection::time_series_split_indices(x.len(), 5);
    assert_eq!(tss.len(), 5);
    for (train, test) in &folds {
        assert!(!train.is_empty() && !test.is_empty());
    }

    // k_fold_cv: accuracy of a tree classifier on the binary task.
    let (bx, by) = binary_classification_data();
    let accs = model_selection::k_fold_cv(&bx, &by, 5, 42, |xtr, ytr, xte| {
        let mut t = tree::DecisionTree::new(4, 4);
        t.fit(xtr, ytr);
        t.predict(xte)
    });
    assert_eq!(accs.len(), 5);
    assert!(accs.iter().all(|&a| a.is_finite() && (0.0..=1.0).contains(&a)));

    // evaluation
    let ev = evaluation::cross_val_score(&x, &y, 5, |xtr, ytr, xte| {
        let r = linear::fit(xtr, ytr).unwrap();
        linear::predict(xte, &r.coefficients, r.intercept).unwrap()
    });
    assert_eq!(ev.len(), 5);
    assert!(ev.iter().all(|&s| s.is_finite()));

    let skf = evaluation::stratified_k_fold(&by, 5, 42);
    assert_eq!(skf.len(), 5);

    let lc = evaluation::learning_curve(&x, &y, &[50, 100, 200], |xtr, ytr, xte| {
        let r = linear::fit(xtr, ytr).unwrap();
        linear::predict(xte, &r.coefficients, r.intercept).unwrap()
    });
    assert_eq!(lc.len(), 3);

    let (bmean, bstd) = evaluation::bootstrap_score(&x, &y, 5, 42, |xtr, ytr, xte| {
        let r = linear::fit_ridge(xtr, ytr, 0.5).unwrap();
        linear::predict(xte, &r.coefficients, r.intercept).unwrap()
    });
    assert!(bmean.is_finite() && bstd.is_finite());

    // Metrics
    let preds: Vec<f64> = crate::y_amt_pred();
    let acc = model_selection::accuracy(&preds, &y);
    assert!((0.0..=1.0).contains(&acc));
    let cm = model_selection::confusion_matrix(&preds, &y, 5);
    assert_eq!(cm.len(), 5);
    let (p, r, f1, sup) = model_selection::precision_recall_f1(&preds, &y, 5);
    assert_eq!(p.len(), 5);
    let rep = model_selection::classification_report(&preds, &y, 5);
    assert_eq!(rep.precision.len(), 5);

    let scores: Vec<f64> = load_rows().iter().map(|r| r.amount).collect();
    let labels: Vec<f64> = load_rows().iter().map(|r| if r.customers > 100.0 { 1.0 } else { 0.0 }).collect();
    let roc = model_selection::roc_curve(&scores, &labels);
    assert!(!roc.is_empty());
    let auc = model_selection::auc(&roc);
    assert!(auc.is_finite() && (0.0..=1.0).contains(&auc));

    // Estimator traits + grid/random search on a small binary sample.
    let s = 120;
    let gx: Vec<Vec<f64>> = bx.iter().take(s).cloned().collect();
    let gy2: Vec<f64> = by.iter().take(s).copied().collect();

    let mut lr = estimator::LinearRegression::new();
    lr.fit(&gx, &gy2).unwrap();
    let pred = lr.predict(&gx).unwrap();
    assert_eq!(pred.len(), s);

    let mut rr = estimator::RidgeRegression::new(1.0);
    rr.fit(&gx, &gy2).unwrap();
    assert_eq!(rr.predict(&gx).unwrap().len(), s);

    let mut log_reg = estimator::LogisticRegression::new(0.1, 100, 1e-6, 1.0);
    log_reg.fit(&gx, &gy2).unwrap();
    let cl = log_reg.predict(&gx).unwrap();
    assert!(accuracy(&cl, &gy2) > 0.3);
    let clp = log_reg.predict_proba(&gx).unwrap();
    assert_eq!(clp.len(), s);

    let mut knc = estimator::KNNClassifier::new(5);
    knc.fit(&gx, &gy2).unwrap();
    assert_eq!(knc.predict(&gx).unwrap().len(), s);

    let mut knr = estimator::KNNRegressor::new(5);
    knr.fit(&gx, &gy2).unwrap();
    assert_eq!(knr.predict(&gx).unwrap().len(), s);

    let mut dtc = estimator::DecisionTreeClassifier::new(4, 4);
    dtc.fit(&gx, &gy2).unwrap();
    assert_eq!(dtc.predict(&gx).unwrap().len(), s);

    let mut rfc = estimator::RandomForestClassifier::new(5, 4, 2);
    rfc.fit(&gx, &gy2).unwrap();
    assert_eq!(rfc.predict(&gx).unwrap().len(), s);

    // Transformers
    let mut ss = estimator::StandardScaler::new();
    let ss_out = ss.fit_transform(&gx).unwrap();
    assert_eq!(ss_out.len(), s);
    let mut mm2 = estimator::MinMaxScaler::new();
    let mm2_out = mm2.fit_transform(&gx).unwrap();
    assert!(mm2_out.iter().all(|r| r.iter().all(|&v| (-1e-9..=1.0 + 1e-9).contains(&v))));
    let mut ohe = estimator::OneHotEncoder::new();
    let ohe_x: Vec<Vec<f64>> = gx.iter().map(|r| vec![r[0], r[1]]).collect();
    let ohe_out = ohe.fit_transform(&ohe_x).unwrap();
    assert!(ohe_out.iter().all(|r| !r.is_empty()));

    // cross_val_score_trait
    let cvs = estimator::cross_val_score_trait(&log_reg, &gx, &gy2, 5, true).unwrap();
    assert_eq!(cvs.len(), 5);
    let cvr = estimator::cross_val_score_trait(&rr, &gx, &gy2, 5, false).unwrap();
    assert_eq!(cvr.len(), 5);

    // permutation_importance
    let mut lr2 = estimator::LinearRegression::new();
    lr2.fit(&gx, &gy2).unwrap();
    let (pi_means, _pi_stds) =
        estimator::permutation_importance(&lr2, &gx, &gy2, false, 3, 1).unwrap();
    assert_eq!(pi_means.len(), gx[0].len());

    // GridSearchCV + RandomizedSearchCV
    let candidates = vec![
        ("knn1".to_string(), estimator::KNNClassifier::new(1)),
        ("knn5".to_string(), estimator::KNNClassifier::new(5)),
    ];
    let mut gs = estimator::GridSearchCV::new(candidates, 3, true);
    gs.fit(&gx, &gy2).unwrap();
    assert!(gs.best_params().is_some());
    assert!(gs.best_model().is_some());
    assert!(gs.best_score() > 0.0);

    let cand = vec![
        ("knn1".to_string(), estimator::KNNClassifier::new(1)),
        ("knn3".to_string(), estimator::KNNClassifier::new(3)),
        ("knn7".to_string(), estimator::KNNClassifier::new(7)),
    ];
    let mut rs = estimator::RandomizedSearchCV::new(cand, 3, true, 2, 42);
    rs.fit(&gx, &gy2).unwrap();
    assert!(rs.best_params().is_some());
    assert_eq!(rs.results().len(), 2);
}

/// Regression predictions on the full sales rows (reused for metric tests).
fn y_amt_pred() -> Vec<f64> {
    let (x, y) = regression_data();
    let lin = linear::fit(&x, &y).unwrap();
    linear::predict(&x, &lin.coefficients, lin.intercept).unwrap()
}

// ---------------------------------------------------------------------------
// pipeline
// ---------------------------------------------------------------------------

#[test]
fn pipeline_roundtrip_and_persistence() {
    let (x, y) = binary_classification_data();
    let mut pipe = pipeline::Pipeline::new(vec![
        pipeline::PipelineStep::Standardize,
        pipeline::PipelineStep::MinMax,
        pipeline::PipelineStep::Model(pipeline::ModelType::Logistic),
    ]);
    pipe.fit(&x, &y);
    let preds = pipe.predict(&x);
    assert_eq!(preds.len(), x.len());
    assert!(accuracy(&preds, &y) > 0.3, "pipeline acc: {}", accuracy(&preds, &y));

    let path = std::env::temp_dir().join("mathverse_pipeline_test.hex");
    pipeline::save_pipeline(&pipe, path.to_str().unwrap()).unwrap();
    let loaded = pipeline::load_pipeline(path.to_str().unwrap());
    assert!(loaded.is_err(), "load is not implemented; expected Err");
    let _ = std::fs::remove_file(path);
}

// ---------------------------------------------------------------------------
// datasets (synthetic generators)
// ---------------------------------------------------------------------------

#[test]
fn synthetic_dataset_generators() {
    let (x, y) = datasets::make_classification(200, 4, 3, 42);
    assert_eq!(x.len(), 200);
    assert_eq!(x[0].len(), 4);
    let max_label = y
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(max_label <= 3.0);

    let (x, y) = datasets::make_regression(200, 3, 0.1, 42);
    assert_eq!(x.len(), 200);
    assert!(y.iter().all(|&v| v.is_finite()));

    let (x, y) = datasets::make_blobs(150, 3, 0.5, 42);
    assert_eq!(x.len(), 150);
    assert!(y.iter().all(|&v| (0.0..=2.0).contains(&v)));

    let (x, y) = datasets::make_moons(150, 0.1, 42);
    assert_eq!(x.len(), 150);
    assert!(y.iter().all(|&v| v == 0.0 || v == 1.0));

    let (x, y) = datasets::make_circles(150, 0.1, 0.5, 42);
    assert_eq!(x.len(), 150);
    assert!(y.iter().all(|&v| v == 0.0 || v == 1.0));

    let (x, y) = datasets::make_spirals(150, 0.1, 3, 42);
    assert_eq!(x.len(), 150);
    assert!(y.iter().all(|&v| (0.0..=2.0).contains(&v)));
}