"""
Ana Fraud Detection Pipeline
Training, inference ve deployment icin end-to-end pipeline
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
import logging
import argparse
from datetime import datetime
import warnings

# Local imports
# NOT: Bu modullerin (src.preprocessing, src.evaluation, src.outlier_detection)
# projenizin src/ klasorunde mevcut oldugunu varsayar.
from src.preprocessing import FeaturePreprocessor, ImbalanceHandler
from src.evaluation import FraudEvaluator

try:
    # Bu modulun projenizde mevcut oldugundan emin olun.
    from explainability_clean import ModelExplainer
except ImportError:
    print("⚠️ Explainability module import hatasi")
    ModelExplainer = None


from src.outlier_detection import OutlierDetector
warnings.filterwarnings('ignore')

# Global logger initialization
logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    """End-to-end Fraud Detection Pipeline"""
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Args:
            config_path (str): Konfigurasyon dosyasi yolu
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_mlflow()
        
        # Pipeline components
        self.preprocessor = None
        self.models = {}
        self.evaluators = {}
        self.explainer = None
        
        # Data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Processed/Balanced Data
        self.X_train_processed = None
        self.y_train_processed = None
        self.X_test_processed = None
        self.y_test_processed = None
        self.X_train_balanced = None
        self.y_train_balanced = None
        
        logger.info("Fraud Detection Pipeline initialized")
    
    def _load_config(self, config_path):
        """Konfigurasyon dosyasini yukle"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Default konfigurasyon"""
        return {
            'data': {'test_size': 0.3, 'random_state': 42, 'stratify': True},
            'preprocessing': {'scaling_method': 'robust', 'encoding_method': 'onehot'},
            'imbalance': {'method': 'smote', 'sampling_strategy': 'auto', 'random_state': 42},
            'models': {
                'random_forest': {'n_estimators': 100, 'random_state': 42},
                'logistic_regression': {'random_state': 42, 'solver': 'liblinear'},
                'isolation_forest': {'contamination': 0.05, 'random_state': 42}
            },
            'evaluation': {'min_roc_auc': 0.7, 'min_pr_auc': 0.3},
            'logging': {'level': 'INFO', 'file_logging': False},
            'mlflow': {'tracking_uri': 'sqlite:///mlflow.db', 'experiment_name': 'fraud_detection', 'autolog': {'sklearn': True}}
        }
    
    def _setup_logging(self):
        """Logging kurulumu"""
        global logger
        
        log_config = self.config.get('logging', {})
        log_level_str = log_config.get('level', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO) 
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Reset existing handlers to avoid duplicate logs in case of re-initialization
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # Console logging
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(stream_handler)
        logger.setLevel(log_level)
        
        # File logging
        if log_config.get('file_logging', False):
            os.makedirs('logs', exist_ok=True)
            file_handler = logging.FileHandler(log_config.get('log_file', 'logs/pipeline.log'))
            file_handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(file_handler)
    
    def _setup_mlflow(self):
        """MLflow kurulumu"""
        mlflow_config = self.config.get('mlflow', {})
        
        # Set tracking URI
        tracking_uri = mlflow_config.get('tracking_uri', 'sqlite:///mlflow.db')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        experiment_name = mlflow_config.get('experiment_name', 'fraud_detection')
        mlflow.set_experiment(experiment_name)
        
        # Auto-logging
        if mlflow_config.get('autolog', {}).get('sklearn', True):
            # is_autolog_enabled kontrolü, eski MLflow surumlerinde olmadigi icin kaldirildi.
            # autolog'u dogrudan cagiriyoruz.
            mlflow.sklearn.autolog()
        
        logger.info(f"MLflow configured - Experiment: {experiment_name}")
    
    def _generate_synthetic_data(self, n_samples=5000):
        """Synthetic fraud data olustur"""
        np.random.seed(self.config['data']['random_state'])
        
        # Normal transactions (95%)
        n_normal = int(n_samples * 0.95)
        normal_data = {
            'Amount': np.random.lognormal(2, 1, n_normal),
            'Time': np.random.randint(0, 86400, n_normal),
            'Merchant_Category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail'], n_normal),
            'Customer_Age': np.random.randint(18, 80, n_normal),
            'Is_Weekend': np.random.choice([0, 1], n_normal, p=[0.7, 0.3]),
            'Transaction_Count_Day': np.random.poisson(3, n_normal),
            'Class': np.zeros(n_normal)
        }
        
        # Fraud transactions (5%)
        n_fraud = n_samples - n_normal
        fraud_data = {
            'Amount': np.random.lognormal(4, 2, n_fraud),  # Higher amounts
            'Time': np.random.randint(0, 86400, n_fraud),
            'Merchant_Category': np.random.choice(['online', 'atm', 'international'], n_fraud),
            'Customer_Age': np.random.randint(25, 60, n_fraud),
            'Is_Weekend': np.random.choice([0, 1], n_fraud, p=[0.4, 0.6]),  # More weekend fraud
            'Transaction_Count_Day': np.random.poisson(8, n_fraud),  # More transactions
            'Class': np.ones(n_fraud)
        }
        
        # Combine
        data = {}
        for key in normal_data.keys():
            data[key] = np.concatenate([normal_data[key], fraud_data[key]])
        
        df = pd.DataFrame(data)
        
        # Add some missing values
        missing_indices = np.random.choice(df.index, int(len(df) * 0.02))
        df.loc[missing_indices, 'Customer_Age'] = np.nan
        
        # Shuffle for realistic split
        df = df.sample(frac=1, random_state=self.config['data']['random_state']).reset_index(drop=True)
        
        return df

    def _validate_data(self, data):
        """Data dogrulama"""
        validation_config = self.config.get('data', {}).get('validation', {})
        
        # Required columns
        required_cols = validation_config.get('required_columns', ['Amount', 'Time', 'Class', 'Customer_Age', 'Merchant_Category']) 
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Eksik gerekli sutunlar: {missing_cols}")
        
        # Class values
        if 'Class' in data.columns:
            valid_classes = validation_config.get('class_values', [0, 1])
            data['Class'] = data['Class'].astype(int) 
            invalid_classes = data['Class'].unique()
            invalid_classes = [c for c in invalid_classes if c not in valid_classes]
            if invalid_classes:
                raise ValueError(f"Gecersiz Class degerleri: {invalid_classes}")
        
        # Amount validation
        if 'Amount' in data.columns:
            min_amount = validation_config.get('amount_min', 0)
            max_amount = validation_config.get('amount_max', 1000000)
            if data['Amount'].min() < min_amount or data['Amount'].max() > max_amount:
                logger.warning(f"Amount degerleri beklenen araligin disinda [{min_amount}, {max_amount}]")
        
        logger.info("Data dogrulama tamamlandi")
    
    def load_data(self, data_path=None, synthetic=True, download_with_kagglehub=False):
        """
        Veri yukleme
        
        Args:
            data_path (str): Veri dosyasi yolu
            synthetic (bool): Synthetic data kullan
            download_with_kagglehub (bool): KaggleHub ile otomatik indirme
        """
        data = None
        if download_with_kagglehub:
            logger.info("KaggleHub ile Credit Card Fraud dataset indiriliyor...")
            try:
                import kagglehub
                # NOT: Bu asama Kaggle API anahtarinizin kurulu olmasini gerektirir.
                path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
                csv_file = os.path.join(path, "creditcard.csv")
                if os.path.exists(csv_file):
                    data = pd.read_csv(csv_file)
                    logger.info(f"KaggleHub dataset yuklendi: {data.shape}")
                else:
                    logger.warning("KaggleHub'dan CSV bulunamadi, synthetic data kullaniliyor")
            except Exception as e:
                logger.error(f"KaggleHub indirme hatasi: {e}, synthetic data kullaniliyor")
        
        if data is None:
             if synthetic or data_path is None:
                logger.info("Synthetic fraud data olusturuluyor...")
                data = self._generate_synthetic_data()
             else:
                logger.info(f"Veri yukleniyor: {data_path}")
                data = pd.read_csv(data_path)
        
        if data is None:
            raise RuntimeError("Veri yuklenemedi veya olusturulamadi.")

        # Data validation
        self._validate_data(data)
        
        # Train-test split
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        # Stratify kontrolu
        stratify_y = y if self.config['data'].get('stratify', True) else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=stratify_y
        )
        
        logger.info(f"Data yuklendi - Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        logger.info(f"Class dagilimi - Train: {np.bincount(self.y_train)}, Test: {np.bincount(self.y_test)}")

    def preprocess_data(self):
        """Data on isleme"""
        logger.info("Data on isleme baslatiliyor...")
        
        preprocessing_config = self.config.get('preprocessing', {})
        
        # Initialize preprocessor
        self.preprocessor = FeaturePreprocessor(
            scaling_method=preprocessing_config.get('scaling_method', 'robust'),
            encoding_method=preprocessing_config.get('encoding_method', 'onehot')
        )
        
        # Preprocess training data
        train_data = pd.concat([self.X_train, self.y_train], axis=1)
        train_processed = self.preprocessor.fit_transform(train_data, target_col='Class')
        
        self.X_train_processed = train_processed.drop('Class', axis=1)
        self.y_train_processed = train_processed['Class']
        
        # Preprocess test data
        test_data = pd.concat([self.X_test, self.y_test], axis=1)
        test_processed = self.preprocessor.transform(test_data)
        
        self.X_test_processed = test_processed.drop('Class', axis=1)
        self.y_test_processed = test_processed['Class']
        
        logger.info(f"On isleme tamamlandi - Feature sayisi: {self.X_train_processed.shape[1]}")
        
        # Handle class imbalance
        imbalance_config = self.config.get('imbalance', {})
        method = imbalance_config.get('method', 'smote').lower()
        
        self.X_train_balanced = self.X_train_processed
        self.y_train_balanced = self.y_train_processed
        
        if method == 'smote':
            try:
                self.X_train_balanced, self.y_train_balanced = ImbalanceHandler.apply_smote(
                    self.X_train_processed, self.y_train_processed,
                    sampling_strategy=imbalance_config.get('sampling_strategy', 'auto'),
                    random_state=imbalance_config.get('random_state', 42)
                )
            except Exception as e:
                 logger.warning(f"SMOTE uygulama hatasi: {e}. Dengeleme yapilmiyor.")
        elif method == 'adasyn':
            try:
                self.X_train_balanced, self.y_train_balanced = ImbalanceHandler.apply_adasyn(
                    self.X_train_processed, self.y_train_processed,
                    random_state=imbalance_config.get('random_state', 42)
                )
            except Exception as e:
                logger.warning(f"ADASYN uygulama hatasi: {e}. Dengeleme yapilmiyor.")
            
        logger.info(f"Class dengeleme tamamlandi ({method.upper()}) - Son Train boyutu: {len(self.X_train_balanced)}")

    def train_models(self):
        """Model egitimi"""
        logger.info("Model egitimi baslatiliyor...")
        
        models_config = self.config.get('models', {})
        
        # Check if balanced data is available
        if self.X_train_balanced is None or self.y_train_balanced is None:
             logger.error("Dengelenmis egitim verisi mevcut degil. Egitim durduruluyor.")
             return

        with mlflow.start_run():
            # Log global parameters
            mlflow.log_params({
                'data_size_balanced': len(self.X_train_balanced),
                'n_features': self.X_train_balanced.shape[1],
                'preprocessing_method': self.config.get('preprocessing', {}).get('scaling_method', 'robust'),
                'imbalance_method': self.config.get('imbalance', {}).get('method', 'none')
            })
            
            for model_name, model_params in models_config.items():
                logger.info(f"Training {model_name}...")
                
                # Initialize model
                model = None
                try:
                    if model_name == 'random_forest':
                        model = RandomForestClassifier(**model_params)
                    elif model_name == 'logistic_regression':
                        model = LogisticRegression(**model_params)
                    elif model_name == 'isolation_forest':
                        model = IsolationForest(**model_params)
    
                        # MLflow çakışmasını önlemek için IF'i ayrı bir nested run içinde eğit
                        # Autologging'i kapatıyoruz ve parametreleri manuel logluyoruz.
                        with mlflow.start_run(nested=True, run_name="IF_Training"):
                            model.fit(self.X_train_processed) 
                            mlflow.log_params(model_params) # Parametreleri kaydet

                        self.models[model_name] = model # Eğitilmiş modeli kaydet
                        continue # <-- BU SATIR ÇOK ÖNEMLİ! Aşağıdaki genel fit bloğunu atlamasını sağlar.
    
                    elif model_name == 'lof':
                        model = LocalOutlierFactor(novelty=True, **model_params)
                    else:
                        logger.warning(f"Bilinmeyen model: {model_name}. Atlaniliyor.")
                        continue
                except TypeError as e:
                    logger.error(f"{model_name} model parametre hatasi: {e}. Model atlaniliyor.")
                    continue
                
                # Train model
                if model_name in ['lof']:
                    # Unsupervised models, fit only on features (preferably unbalanced/raw processed data)
                    model.fit(self.X_train_processed) 
                else:
                    # Supervised models (uses balanced data if applied)
                    model.fit(self.X_train_balanced, self.y_train_balanced)
                
                self.models[model_name] = model
                
                # Cross validation for supervised models
                if model_name not in ['isolation_forest', 'lof']:
                    try:
                        cv_scores = cross_val_score(
                            model, self.X_train_balanced, self.y_train_balanced,
                            cv=StratifiedKFold(n_splits=5), scoring='roc_auc', n_jobs=-1
                        )
                        logger.info(f"{model_name} CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                    except Exception as e:
                        logger.warning(f"{model_name} Cross Validation hatasi: {e}")
                
                logger.info(f"{model_name} egitimi tamamlandi")
            
            logger.info("Tum modeller basariyla egitildi")

    def evaluate_models(self):
        """Model degerlendirme"""
        logger.info("Model degerlendirme baslatiliyor...")
        
        if self.X_test_processed is None or self.y_test_processed is None:
            logger.error("Test verisi mevcut degil. Degerlendirme durduruluyor.")
            return
            
        evaluation_config = self.config.get('evaluation', {})
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            
            # Initialize evaluator
            evaluator = FraudEvaluator(model, model_name)
            
            # Evaluate
            try:
                if model_name in ['isolation_forest', 'lof']:
                    # Outlier detection evaluation
                    
                    # Isolation Forest: -1 Outlier (Fraud), 1 Inlier (Normal)
                    predictions_raw = model.predict(self.X_test_processed)
                    predictions = np.where(predictions_raw == -1, 1, 0) # 1:Fraud, 0:Normal
                    
                    # decision_function/score_samples (lower score is more anomalous/fraudulent)
                    scores = model.decision_function(self.X_test_processed)
                    
                    # ROC-AUC ve PR-AUC icin raw skorlar kullaniliyor
                    results = evaluator.evaluate_binary_classification(
                    self.X_test_processed, self.y_test_processed,
                    y_pred_proba=-scores
                    )

                    
                    # Unsupervised modellerde, FraudEvaluator icinde roc_auc hesaplanirken 
                    # skorlarin ters cevrilmesi gerekiyorsa (daha kucuk skor = daha yuksek olasilik), 
                    # FraudEvaluator bu durumu iceride yonetmelidir. 
                    # Burada y_pred_proba olarak scores (-1, 1 araliginda) gonderildi.

                else:
                    # Supervised model evaluation
                    results = evaluator.evaluate_binary_classification(
                        self.X_test_processed, self.y_test_processed
                    )
            except Exception as e:
                logger.error(f"{model_name} degerlendirme hatasi: {e}")
                continue

            self.evaluators[model_name] = evaluator
            
            # Log metrics to MLflow
            with mlflow.start_run(nested=True, run_name=f"Evaluation_{model_name}"):
                if model_name in ['isolation_forest', 'lof']:
                    # Sadece unsupervised modellerin parametrelerini manuel logla
                    if hasattr(model, 'get_params'):
                        mlflow.log_params({f"{model_name}_params_{k}": v for k, v in model.get_params().items()})
                    
                mlflow.log_metrics({
                    f"{model_name}_test_roc_auc": results.get('roc_auc', 0),
                    f"{model_name}_test_pr_auc": results.get('pr_auc', 0),
                    f"{model_name}_test_f1_score": results.get('f1_score', 0),
                    f"{model_name}_test_precision": results.get('precision', 0),
                    f"{model_name}_test_recall": results.get('recall', 0)
                })
                
                # Log model (sadece supervised modelleri kaydediyoruz)
                if model_name not in ['isolation_forest', 'lof']:
                    mlflow.sklearn.log_model(model, f"{model_name}_model")
            
            # Print evaluation report
            logger.info("Model Evaluation Summary:")
            logger.info(evaluator.results) 
            
            # Check performance thresholds
            min_roc_auc = evaluation_config.get('min_roc_auc', 0.7)
            min_pr_auc = evaluation_config.get('min_pr_auc', 0.3)
            
            if results.get('roc_auc', 0) < min_roc_auc:
                logger.warning(f"{model_name} ROC-AUC ({results.get('roc_auc', 0):.4f}) esik degerin altinda ({min_roc_auc})")
            
            if results.get('pr_auc', 0) < min_pr_auc:
                logger.warning(f"{model_name} PR-AUC ({results.get('pr_auc', 0):.4f}) esik degerin altinda ({min_pr_auc})")
        
        logger.info("Model degerlendirme tamamlandi")

    def _find_best_model(self):
        """En iyi modeli bul (ROC-AUC'ye gore)"""
        best_model = None
        best_score = -np.inf 
        
        # Supervised modeller arasindan en iyi ROC-AUC'yi bul
        for model_name, evaluator in self.evaluators.items():
            if model_name in ['isolation_forest', 'lof']:
                # Unsupervised modelleri, supervised metrikle karsilastirilmadigi icin atla
                continue 
                
            if evaluator.results and 'roc_auc' in evaluator.results:
                roc_auc = evaluator.results['roc_auc']
                if roc_auc > best_score:
                    best_score = roc_auc
                    best_model = model_name
        
        best_model = best_model or 'random_forest'
        
        logger.info(f"Best model: {best_model} (ROC-AUC: {best_score:.4f})")
        return best_model 

    def explain_models(self, model_name='random_forest'):
        """Model aciklanabilirligi"""
        if ModelExplainer is None:
            logger.warning("ModelExplainer mevcut degil - aciklanabilirlik atlaniliyor")
            return None, None
            
        if model_name not in self.models:
            logger.error(f"Model {model_name} bulunamadi")
            return None, None
        
        if self.X_train_balanced is None or self.X_test_processed is None:
            logger.error("Egitim/Test verisi islenmemis veya yuklenmemis. Aciklanabilirlik durduruluyor.")
            return None, None
        
        # Unsupervised modelleri atla
        if model_name in ['isolation_forest', 'lof']:
            logger.warning(f"Model ({model_name}) unsupervised. Aciklanabilirlik atlaniliyor.")
            return None, None
            
        logger.info(f"Explaining {model_name}...")
        
        try:
            # Initialize explainer
            self.explainer = ModelExplainer(
                self.models[model_name],
                self.X_train_balanced,
                feature_names=list(self.X_train_processed.columns),
                class_names=['Normal', 'Fraud']
            )
            
            # SHAP analysis
            explainer_config = self.config.get('explainability', {}).get('shap', {})
            self.explainer.initialize_shap(
                explainer_type=explainer_config.get('explainer_type', 'auto'),
                max_evals=explainer_config.get('max_evals', 100)
            )
            
            # Use a sample of the test set for SHAP calculation
            max_samples = explainer_config.get('max_samples', 100)
            shap_values, X_sample = self.explainer.compute_shap_values(
                self.X_test_processed,
                max_samples=max_samples
            )
            
            # Generate plots (assuming ModelExplainer handles saving/display)
            self.explainer.plot_shap_summary(X_sample)
            self.explainer.plot_shap_waterfall(X_sample, 0)
            
            # Global feature importance
            importance = self.explainer.global_feature_importance(X_sample)
            
            # Fraud pattern analysis
            y_sample = self.y_test_processed.iloc[:len(X_sample)]
            fraud_patterns = self.explainer.analyze_fraud_patterns(
                X_sample, y_sample
            )
            
            logger.info("Model aciklamasi tamamlandi")
            return importance, fraud_patterns
            
        except Exception as e:
            logger.error(f"Aciklanabilirlik hatasi: {e}")
            return None, None

    def save_models(self, save_path="models/"):
        """Model ve preprocessor kaydetme"""
        os.makedirs(save_path, exist_ok=True)
        
        if self.preprocessor is None:
            logger.warning("Preprocessor mevcut degil, kaydedilmiyor.")
        else:
            # Save preprocessor
            joblib.dump(self.preprocessor, os.path.join(save_path, 'preprocessor.pkl'))
            logger.info("Preprocessor kaydedildi")
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(save_path, f'{model_name}_model.pkl')
            joblib.dump(model, model_path)
            logger.info(f"{model_name} model kaydedildi: {model_path}")
        
        # Save feature names
        if self.X_train_processed is not None:
            feature_info = {
                'feature_names': list(self.X_train_processed.columns),
                'n_features': len(self.X_train_processed.columns),
                'preprocessing_config': self.config.get('preprocessing', {})
            }
            joblib.dump(feature_info, os.path.join(save_path, 'feature_info.pkl'))
        
        logger.info(f"Tum modeller {save_path} klasorune kaydedildi")

    def load_models(self, load_path="models/"):
        """Model ve preprocessor yukleme"""
        if not os.path.exists(load_path):
            logger.error(f"Model dizini bulunamadi: {load_path}")
            return False
            
        try:
            # Load preprocessor
            preprocessor_path = os.path.join(load_path, 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info("Preprocessor yuklendi")
            else:
                logger.warning("Preprocessor dosyasi bulunamadi.")
            
            # Load models
            model_files = [f for f in os.listdir(load_path) if f.endswith('_model.pkl')]
            if not model_files:
                logger.warning("Kaydedilmis model dosyasi bulunamadi.")
            
            for model_file in model_files:
                model_name = model_file.replace('_model.pkl', '')
                model_path = os.path.join(load_path, model_file)
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"{model_name} model yuklendi")
            
            # Load feature info
            feature_info_path = os.path.join(load_path, 'feature_info.pkl')
            if os.path.exists(feature_info_path):
                feature_info = joblib.load(feature_info_path)
                logger.info(f"Feature bilgisi yuklendi - {feature_info['n_features']} features")
            
            logger.info(f"Tum modeller {load_path} klasorunden yuklendi")
            return True
            
        except Exception as e:
            logger.error(f"Model yukleme hatasi: {e}")
            return False
    
    def predict(self, data, model_name='random_forest'):
        """Tahmin (Inference)"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} bulunamadi")
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor yuklenmedi")
        
        # Preprocess data
        data_processed = self.preprocessor.transform(data.copy()) 
        
        # Make prediction
        model = self.models[model_name]
        
        if model_name in ['isolation_forest', 'lof']:
            # Outlier detection
            # -1 (Outlier/Fraud), 1 (Inlier/Normal)
            predictions_raw = model.predict(data_processed)
            predictions = np.where(predictions_raw == -1, 1, 0) # 1:Fraud, 0:Normal
            
            # Decision function (lower score is more anomalous/fraudulent)
            scores = model.decision_function(data_processed)
            
            # Scores'u ters cevirerek olasilik gibi kullanmak icin basit min-max scaling
            min_score = np.min(scores)
            max_score = np.max(scores)
            # 1.0 - [ (score - min_score) / (max_score - min_score) ]
            probabilities = 1.0 - (scores - min_score) / (max_score - min_score + 1e-6) 
            
            return predictions, probabilities
        else:
            # Supervised model
            predictions = model.predict(data_processed)
            # Tahmin olasiliklari (pozitif sinif (Fraud) olasiligi)
            probabilities = model.predict_proba(data_processed)[:, 1]
            return predictions, probabilities

    def run_full_pipeline(self, data_path=None, save_models=True, use_kagglehub=False):
        """Full pipeline calistirma"""
        logger.info("Full Fraud Detection Pipeline baslatiliyor...")
        
        try:
            # 1. Load data
            self.load_data(data_path, synthetic=(data_path is None and not use_kagglehub), 
                           download_with_kagglehub=use_kagglehub)
            
            # 2. Preprocess
            self.preprocess_data()
            
            # 3. Train models
            self.train_models()
            
            # 4. Evaluate models
            self.evaluate_models()
            
            # 5. Explain best model
            best_model = self._find_best_model()
            # Explainability sadece supervised modellerde calisir.
            if best_model not in ['isolation_forest', 'lof']:
                 self.explain_models(best_model)
            else:
                 logger.warning(f"Best model ({best_model}) unsupervised, explainability atlaniliyor.")
            
            # 6. Save models
            if save_models:
                self.save_models()
            
            logger.info("Full pipeline basariyla tamamlandi!")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline hatasi: {e}", exc_info=True)
            return False


def main():
    """CLI arayuzu"""
    parser = argparse.ArgumentParser(description='Fraud Detection Pipeline')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--data', help='Data file path (optional, uses synthetic if not provided)')
    parser.add_argument('--mode', choices=['train', 'predict', 'explain'], default='train', help='Pipeline modu')
    parser.add_argument('--model', default='random_forest', help='Tahmin/Aciklama icin model adi')
    parser.add_argument('--load_models', action='store_true', help='Mevcut modelleri yukle')
    parser.add_argument('--save_models', action='store_true', help='Egitilmis modelleri kaydet')
    parser.add_argument('--use_kagglehub', action='store_true', help='KaggleHub ile veri indir')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FraudDetectionPipeline(args.config)
    
    if args.mode == 'train':
        # Training mode
        success = pipeline.run_full_pipeline(args.data, args.save_models, args.use_kagglehub)
        sys.exit(0 if success else 1)
        
    elif args.mode == 'predict':
        # Prediction mode
        if args.load_models:
            if not pipeline.load_models():
                logger.error("Model yuklenemedi. Tahmin yapilamiyor.")
                sys.exit(1)

        try:
            # Veri yukle (ham veriyi alabilmek icin)
            pipeline.load_data(data_path=args.data, synthetic=(args.data is None), download_with_kagglehub=args.use_kagglehub)
            
            # Eğer load_models yapıldıysa, pipeline.predict otomatik olarak preprocessor'ı kullanacaktır.
            # Ancak train modu calismadiysa X_test_processed/X_test tanimli degildir.
            if pipeline.X_test is None:
                pipeline.preprocess_data() # Sadece preprocessor'ı olusturmak ve veriyi bolmek icin
            
            if not pipeline.models:
                logger.warning("Model egitilmemis ve yuklenmemis. Tahmin icin once train modunu calistirin.")
                sys.exit(1)

            # Raw test verisinin ilk 5 ornegi uzerinde tahmin yap (predict icinde islenecektir)
            raw_X_test_sample = pipeline.X_test.head()

            predictions, probabilities = pipeline.predict(
                raw_X_test_sample, args.model
            )
            
            print("\nSample Predictions:")
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                print(f"Sample {i}: Prediction={int(pred)}, Fraud Probability={prob:.4f}")
                
        except Exception as e:
            logger.error(f"Tahmin modu hatasi: {e}")
            sys.exit(1)
            
    elif args.mode == 'explain':
        # Explanation mode
        if args.load_models:
            if not pipeline.load_models():
                logger.error("Model yuklenemedi. Aciklama yapilamiyor.")
                sys.exit(1)

        try:
            # Veri yukle ve isleme (Aciklama icin gereklidir)
            pipeline.load_data(data_path=args.data, synthetic=(args.data is None), download_with_kagglehub=args.use_kagglehub)
            pipeline.preprocess_data()
            
            if args.model not in pipeline.models:
                logger.error(f"Aciklama icin istenen model ({args.model}) mevcut degil. Train modunu calistirdiginizdan emin olun.")
                sys.exit(1)
                
            importance, patterns = pipeline.explain_models(args.model)
            
            if importance is not None:
                print("\nTop 10 Onemli Feature (Global SHAP):")
                sorted_importance = sorted(importance.items(), key=lambda item: item[1], reverse=True)
                for i, (feature, score) in enumerate(sorted_importance[:10]):
                    print(f"{i+1}. {feature}: {score:.4f}")
            
            if patterns is not None:
                print("\nFraud Pattern Analizi (Ozet):")
                print(patterns) 

        except Exception as e:
            logger.error(f"Aciklama modu hatasi: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()