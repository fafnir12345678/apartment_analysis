import os
import sys
import traceback
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.patches import Patch

from catboost import CatBoostRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import statsmodels.api as sm
from scipy import stats
from scipy.stats import f as f_dist, gaussian_kde
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_breuschpagan,
    het_goldfeldquandt,
)
from statsmodels.stats.outliers_influence import reset_ramsey, variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

import tkinter as tk
from tkinter import ttk, messagebox


# ==============================================================================
# 0. КОНФИГУРАЦИЯ И КОНСТАНТЫ
# ==============================================================================

CONFIG = {
    "DATA_PATH_INPUT": os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_data"),
    "DATA_PATH_OUTPUT": os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_data"),
    "TARGET_COL": "price",
    "TEST_SIZE": 0.2,
    "RANDOM_STATE": 42,
    "DROP_FEATURES": ["link", "construction_year", "price", "region_original"],
    "ALPHA": 0.01,
    "APARTMENT_INDEX": 100,
    "CORR_THRESHOLD": 0.7,
    "VIF_THRESHOLD": 10,
    "COOKS_FACTOR": 4,
    "priority_threshold": 2
}

# ==============================================================================
# 1. БАЗОВЫЕ УТИЛИТЫ И РАСЧЁТ МЕТРИК (не зависят от данных/моделей)
# ==============================================================================

def huber_loss_func(
    y_t: np.ndarray, 
    y_p: np.ndarray, 
    delta: float = 1e6
    ) -> float:
    """Вычисляет функцию потерь Хьюбера, устойчивую к выбросам.
    
    Комбинирует квадратичную ошибку для малых отклонений и линейную для больших,
    что снижает влияние аномальных значений на итоговую метрику.
    
    Args:
        y_t: Массив фактических значений целевой переменной.
        y_p: Массив предсказанных значений.
        delta: Пороговое значение, определяющее границу перехода от квадратичной 
               к линейной функции потерь.
               
    Returns:
        Среднее значение функции потерь Хьюбера.
    """
    error = y_t - y_p
    is_small_error = np.abs(error) <= delta
    return np.mean(np.where(is_small_error, 0.5 * error**2, delta * (np.abs(error) - 0.5 * delta)))

def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    model_obj: Any = None,
    n_features: Optional[int] = None,
    model_obj_log: Any = None,
    y_log_true: Optional[np.ndarray] = None,
    y_log_pred: Optional[np.ndarray] = None,
    silent: bool = False
) -> Dict[str, Any]:
    """Рассчитывает комплекс метрик качества регрессионной модели.
    
    Включает базовые ошибки (MAE, MSE, RMSE, MAPE, MSLE), коэффициент детерминации,
    скорректированный R2 (с учётом линейности модели), а также статистику остатков
    (асимметрия, эксцесс, оценка нормальности).
    
    Args:
        y_true: Фактические значения целевой переменной.
        y_pred: Предсказанные моделью значения.
        model_name: Идентификатор модели для вывода в консоль.
        model_obj: Обученный объект модели (statsmodels/sklearn).
        n_features: Количество признаков (используется для ручного расчёта Adj. R2).
        model_obj_log: Объект логарифмированной модели (если применимо).
        y_log_true: Логарифмированные фактические значения.
        y_log_pred: Логарифмированные предсказанные значения.
        silent: Флаг подавления вывода метрик в стандартный поток.
        
    Returns:
        Словарь с рассчитанными метриками и статусом нормальности остатков.
    """
    y_pred_clipped = np.maximum(y_pred, 0)
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) * 100
    msle = mean_squared_log_error(np.maximum(y_true, 0), y_pred_clipped)
    huber = huber_loss_func(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    non_linear_keywords = ['forest', 'catboost', 'xgb', 'tree', 'boosting', 'gradient', 'random']
    is_non_linear = any(key in model_name.lower() for key in non_linear_keywords)
    
    adj_r2 = np.nan
    adj_r2_msg = None

    if is_non_linear:
        adj_r2_msg = "— (не применимо для нелинейных моделей)*"
    elif model_obj is not None and hasattr(model_obj, 'rsquared_adj'):
        adj_r2 = model_obj.rsquared_adj
    elif n_features is not None:
        n_samples = len(y_true)
        p_params = n_features
        if n_samples > p_params + 1:
            adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - p_params - 1)
        else:
            adj_r2_msg = f"— (не рассчитан: n={n_samples} ≤ p+1={p_params+1})"
    else:
        adj_r2_msg = "— (отсутствуют параметры модели)"

    r2_log = np.nan
    adj_r2_log = np.nan
    is_log_model = False
    
    if model_obj_log is not None and hasattr(model_obj_log, 'rsquared'):
        is_log_model = True
        r2_log = model_obj_log.rsquared
        adj_r2_log = model_obj_log.rsquared_adj
    elif y_log_true is not None and y_log_pred is not None:
        is_log_model = True
        r2_log = r2_score(y_log_true, y_log_pred)
        if n_features is not None:
            n_log = len(y_log_true)
            p_log = n_features
            if n_log > p_log + 1:
                adj_r2_log = 1 - (1 - r2_log) * (n_log - 1) / (n_log - p_log - 1)

    resid_values = model_obj.resid if model_obj is not None else (y_true - y_pred)
    skewness = stats.skew(resid_values)
    kurtosis = stats.kurtosis(resid_values, fisher=False)
    normality_status = "Нормально распределены" if abs(skewness) < 0.5 and abs(kurtosis - 3) < 1 else "Отклоняются от нормы"

    results = {
        'R2': r2, 
        'Adj_R2': np.nan if is_non_linear else adj_r2,
        'Adj_R2_Msg': adj_r2_msg,
        'R2_log': r2_log if is_log_model else np.nan,
        'Adj_R2_log': adj_r2_log if is_log_model else np.nan,
        'MAE': mae, 'MSE': mse, 'RMSE': rmse,
        'MAPE': mape, 'MSLE': msle, 'Huber': huber,
        'Skewness': skewness, 'Kurtosis': kurtosis,
        'Normality_Status': normality_status,
        'Is_Non_Linear': is_non_linear
    }

    if not silent:
        print_metrics_template(results, model_name)

    return results

def calculate_anova(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    n_obs: int, 
    n_features: int
) -> Tuple[pd.DataFrame, float, float]:
    """Формирует таблицу дисперсионного анализа (ANOVA) для регрессии.
    
    Разделяет общую вариацию на объяснённую моделью и остаточную, рассчитывает
    F-статистику и её p-value для проверки глобальной значимости уравнения.
    
    Args:
        y_true: Фактические значения.
        y_pred: Предсказанные значения.
        n_obs: Количество наблюдений в выборке.
        n_features: Количество регрессоров.
        
    Returns:
        Кортеж из (DataFrame таблицы ANOVA, сумма квадратов ошибок, средняя ошибка).
    """
    y_mean = np.mean(y_true)
    ssr = np.sum((y_pred - y_mean)**2)
    sse = np.sum((y_true - y_pred)**2)
    sst = ssr + sse
    
    df_ssr = n_features
    df_sse = n_obs - n_features - 1
    df_sst = n_obs - 1
    
    msr = ssr / df_ssr if df_ssr > 0 else 0
    mse_anova = sse / df_sse if df_sse > 0 else 0
    
    F_stat = msr / mse_anova if mse_anova > 0 else 0
    p_value = f_dist.sf(F_stat, df_ssr, df_sse)
    
    anova_table = pd.DataFrame({
        'Источник дисперсии': ['Регрессия (SSR)', 'Ошибки (SSE)', 'Итого (SST)'],
        'Сумма квадратов (SS)': [ssr, sse, sst],
        'Степени свободы (df)': [df_ssr, df_sse, df_sst],
        'Средний квадрат (MS)': [msr, mse_anova, np.nan],
        'F-статистика': [F_stat, np.nan, np.nan],
        'p-value': [p_value, np.nan, np.nan]
    })
    return anova_table, sse, mse_anova

def print_metrics_template(
    metrics: Dict[str, Any],
    model_name: str
    ) -> None:
    """Стандартизированный вывод метрик модели в консоль.
    
    Поддерживает линейные, полиномиальные и ансамблевые модели. 
    Автоматически форматирует денежные величины и обрабатывает отсутствие 
    скорректированного R2 для нелинейных алгоритмов.
    
    Args:
        metrics: Словарь с рассчитанными метриками.
        model_name: Название модели для заголовка.
    """
    def fmt(val: Any, precision: int = 3, is_money: bool = False) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        if is_money:
            return f"{val:,.0f} руб."
        return f"{val:.{precision}f}"

    print(f"\n--- Метрики для {model_name} ---")
    print(f"{model_name} R2 (рубли): {fmt(metrics.get('R2'))}")
    
    adj_msg = metrics.get('Adj_R2_Msg')
    adj_val = metrics.get('Adj_R2')
    if adj_msg:
        print(f"{model_name} Adjusted R2 (рубли): {adj_msg}")
    else:
        print(f"{model_name} Adjusted R2 (рубли): {fmt(adj_val)}")

    r2_log = metrics.get('R2_log')
    if r2_log is not None and not np.isnan(r2_log):
        print(f"{model_name} R2 (log-шкала): {fmt(r2_log)}")
    
    print(f"Средняя абсолютная ошибка (MAE): {fmt(metrics.get('MAE'), is_money=True)}")
    print(f"Среднеквадратичная ошибка (MSE): {metrics.get('MSE', 0):.2e}")
    print(f"Корень из MSE (RMSE): {fmt(metrics.get('RMSE'), is_money=True)}")
    print(f"Средняя процентная ошибка (MAPE): {fmt(metrics.get('MAPE'), 2)}%")
    print(f"Среднеквадратичная логарифмическая ошибка (MSLE): {fmt(metrics.get('MSLE', 0), 4)}")
    print(f"Корень из MSLE (RMSLE): {fmt(metrics.get('MSLE', 0)**0.5, 4)}")
    print(f"Потеря Хьюбера (Huber Loss): {fmt(metrics.get('Huber'), is_money=True)}")
    
    print(f"Асимметрия (Skewness): {fmt(metrics.get('Skewness'))}")
    print(f"Эксцесс (Kurtosis): {fmt(metrics.get('Kurtosis'))}")
    print(f"Статус нормальности: {metrics.get('Normality_Status', '—')}")

    if metrics.get('Is_Non_Linear', False):
        print("\n* Примечание: Для нелинейных моделей (RF, CatBoost, Poly) "
              "Adj. R² носит оценочный характер.")

def _capture_output_and_save(
    file_name: str, 
    capture_func: Callable[[], None]
    ) -> None:
    """Перехватывает вывод stdout и сохраняет его в текстовый файл."""
    old_stdout = sys.stdout
    mystdout = StringIO()
    sys.stdout = mystdout
    
    try:
        capture_func()
    except Exception as e:
        print(f"\n[КРИТИЧЕСКАЯ ОШИБКА ВНУТРИ ГЕНЕРАЦИИ]: {e}")
    finally:
        content = mystdout.getvalue()
        sys.stdout = old_stdout
        mystdout.close()

    if not content.strip():
        print(f"[!] Внимание: Отчет {file_name} пуст. Запись отменена.")
        return

    # ИСПРАВЛЕНО: Создаём все промежуточные папки пути (включая вложенные, например "text")
    file_path = os.path.join(CONFIG["DATA_PATH_OUTPUT"], file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"Файл отчета успешно сформирован: {file_path}")

# ==============================================================================
# 2. ЗАГРУЗКА, ОЧИСТКА И ПРЕДОБРАБОТКА ДАННЫХ
# ==============================================================================

def load_and_init_data(
    ) -> pd.DataFrame:
    """Загружает CSV-файл из конфигурационной директории.
    
    Автоматически выбирает первый доступный файл в папке input_data 
    и читает его с учётом разделителя ';' и десятичной запятой.
    
    Returns:
        Исходный DataFrame с данными из CSV.
    """
    folder_data_path = CONFIG["DATA_PATH_INPUT"]
    
    if not os.path.exists(folder_data_path):
        raise FileNotFoundError(f"Директория '{folder_data_path}' не найдена.")
        
    folder_list = os.listdir(folder_data_path)
    
    if not folder_list:
        raise ValueError(f"В директории '{folder_data_path}' отсутствуют файлы.")

    print("Доступные файлы:")
    for i, file_name in enumerate(folder_list):
        print(f"    {i+1}. {file_name}")
        
    selected_file = folder_list[0] 
    data_csv_path = os.path.join(folder_data_path, selected_file)
    df = pd.read_csv(data_csv_path, sep=";", decimal=",")
    print(f"Файл {selected_file} загружен успешно")
    return df

def filter_by_percentile(
    df: pd.DataFrame,
    column: str = 'price'
    ) -> pd.DataFrame:
    """Удаляет выбросы методом процентилей (1-й и 99-й).
    
    Обрезает хвосты распределения, оставляя только наблюдения, попадающие 
    в межпроцентильный диапазон. Подходит для данных с тяжелыми хвостами.
    
    Args:
        df: DataFrame для фильтрации.
        column: Имя целевого столбца.
        
    Returns:
        Отфильтрованный DataFrame.
    """
    low = df[column].quantile(0.01)
    high = df[column].quantile(0.99)
    df = df[(df[column] >= low) & (df[column] <= high)]
    print(f"Применен метод перцентилей. Границы: {low:.0f} - {high:.0f}")
    return df

def filter_by_iqr(
    df: pd.DataFrame,
    column: str = 'price'
    ) -> pd.DataFrame:
    """Удаляет выбросы методом межквартильного размаха (IQR).
    
    Использует правило 1.5*IQR для определения границ. Более устойчив к 
    асимметричным распределениям, чем процентильный метод.
    
    Args:
        df: DataFrame для фильтрации.
        column: Имя целевого столбца.
        
    Returns:
        Отфильтрованный DataFrame.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = max(0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"Применен метод IQR. Границы: {lower_bound:.0f} - {upper_bound:.0f}")
    return df

def clean_and_prepare_data(
    df: pd.DataFrame
    ) -> pd.DataFrame:
    """Выполняет первичную очистку и кодирование данных.
    
    Удаляет дубликаты, обрабатывает пропуски медианами, создаёт инженерный 
    признак возраста здания, валидирует этажность и кодирует регионы 
    через дамми-переменные с устранением dummy trap.
    
    Args:
        df: Исходный DataFrame.
        
    Returns:
        Очищенный и подготовленный DataFrame.
    """
    df = df.copy()
    df = df.drop_duplicates(subset=["link"], keep="first")
    df = df.dropna(subset=['price'])
    
    if df.empty: 
        return df

    df = filter_by_iqr(df)


    df = df.fillna(df.median(numeric_only=True))

    mode_reg = df['region_of_moscow'].mode()
    if not mode_reg.empty:
        df['region_of_moscow'] = df['region_of_moscow'].fillna(mode_reg.iloc[0])

    df["house_age"] = 2024 - df["construction_year"]
    df = df[(df["house_age"] >= 0) & (df['floor'] <= df['number_of_floors'])]

    df['region_original'] = df['region_of_moscow']
    df = pd.get_dummies(df, columns=['region_of_moscow'], prefix='region_of_moscow', dtype=int)
    
    # Устранение Dummy Trap (удаление базовой категории)
    if 'region_of_moscow_NAR' in df.columns:
        df = df.drop(columns='region_of_moscow_NAR')
    
    return df

def prepare_features(
    df: pd.DataFrame, 
    del_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Разделяет данные на признаки и целевую переменную, формирует train/test.
    
    Args:
        df: Подготовленный DataFrame.
        del_cols: Список колонок для исключения из обучающей выборки.
        
    Returns:
        Кортеж (X_train, X_test, y_train, y_test).
    """
    x_features = df.drop(columns=[col for col in del_cols if col in df.columns])
    y_target = df[CONFIG['TARGET_COL']]
    
    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_target, 
        test_size=CONFIG["TEST_SIZE"], 
        random_state=CONFIG["RANDOM_STATE"]
    )
    return x_train, x_test, y_train, y_test

# ==============================================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ МОДЕЛИРОВАНИЯ (интерпретация, формулы)
# ==============================================================================

def get_model_interpretation(
    model: Any, 
    x_train: pd.DataFrame, 
    y_train: pd.Series, 
    is_log_model: bool = False
    ) -> pd.DataFrame:
    """Рассчитывает эконометрические показатели влияния признаков на целевую переменную.

    Вычисляет стандартизированные коэффициенты (бета-веса), эластичность и 
    процентное влияние для линейных и лог-линейных моделей. Корректно обрабатывает 
    дамми-переменные (бинарные признаки), для которых эластичность не рассчитывается.

    Args:
        model: Обученная модель statsmodels с атрибутами `params` и `pvalues`.
        x_train: Матрица признаков, использованная для обучения.
        y_train: Вектор целевой переменной.
        is_log_model: Флаг, указывающий на использование логарифмированной целевой переменной.

    Returns:
        DataFrame с колонками: Признак, Коэффициент, Бета-коэффициент, P-value, 
        Эластичность (%), Влияние на цену (%).
    """
    has_const = 'const' in model.params
    params = model.params.drop('const') if has_const else model.params
    pvals = model.pvalues.drop('const') if has_const else model.pvalues
    
    df = pd.DataFrame({
        'Признак': x_train.columns,
        'Коэффициент': params.values,
        'Бета-коэффициент': 0.0,
        'P-value': pvals.values,
        'Эластичность (%)': 0.0,
        'Влияние на цену (%)': 0.0
    })

    x_std = x_train.std(ddof=1)
    mean_x = x_train.mean()

    is_dummy = np.array([
        (x_train[col].nunique() == 2 and set(x_train[col].dropna().unique()) <= {0, 1}) or 
        x_train[col].dtype == 'bool' 
        for col in x_train.columns
    ])
    
    if not is_log_model:
        y_std = y_train.std(ddof=1)
        mean_y = y_train.mean()
        
        df['Бета-коэффициент'] = (params * x_std / y_std).fillna(0).values
        if mean_y != 0:
            elasticity_vals = (params * mean_x / mean_y * 100).fillna(0).values
            df.loc[is_dummy, 'Эластичность (%)'] = 0.0
            df.loc[~is_dummy, 'Эластичность (%)'] = elasticity_vals[~is_dummy]
            df['Влияние на цену (%)'] = (params / mean_y * 100).fillna(0).values
    else:
        y_log = np.log1p(y_train)
        y_log_std = y_log.std(ddof=1)
        
        df['Бета-коэффициент'] = (params * x_std / y_log_std).fillna(0).values
        elasticity_vals = (params * mean_x * 100).fillna(0).values

        df.loc[is_dummy, 'Эластичность (%)'] = 0.0
        df.loc[~is_dummy, 'Эластичность (%)'] = elasticity_vals[~is_dummy]

        impact_dummy = (np.exp(params[is_dummy].astype(float)) - 1).values * 100
        impact_cont = (params[~is_dummy] * 100).values
        
        df.loc[is_dummy, 'Влияние на цену (%)'] = impact_dummy
        df.loc[~is_dummy, 'Влияние на цену (%)'] = impact_cont

    return df.fillna(0)

def print_linear_model_equation(
    model: Any,
    feature_names: List[str],
    silent: bool = False
    ) -> str:
    """Формирует строковое представление математического уравнения линейной регрессии.

    Args:
        model: Обученная модель с атрибутами `params`.
        feature_names: Список имён признаков.
        silent: Флаг подавления вывода в консоль.

    Returns:
        Строка с уравнением модели.
    """
    params = model.params
    intercept_coef = params.get('const', 0)
    formula_parts = [f"{intercept_coef:,.2f}"]
    
    for name in feature_names:
        if name in params:
            coef = params[name]
            sign = "+" if coef >= 0 else "-"
            formula_parts.append(f" {sign} ({abs(coef):,.2f} * {name})")
            
    full_formula = "Цена = " + "".join(formula_parts)
    if not silent:
        print("\nМатематическое уравнение модели:")
        print(full_formula)
    return full_formula

def print_log_model_equation(
    model: Any,
    feature_names: List[str],
    silent: bool = False
    ) -> str:
    """Формирует уравнение лог-линейной модели с обратной экспоненциальной трансформацией.

    Args:
        model: Обученная модель.
        feature_names: Список имён признаков.
        silent: Флаг подавления вывода.

    Returns:
        Строка с уравнением вида Цена = exp(...) - 1.
    """
    params = model.params
    intercept_coef = params.get('const', 0)
    parts = [f"{intercept_coef:,.2f}"]
    
    for name in feature_names:
        if name in params:
            coef = params[name]
            sign = "+" if coef >= 0 else "-"
            parts.append(f" {sign} ({abs(coef):.4f} * {name})")

    formula = "Цена = exp(" + "".join(parts) + ") - 1"
    if not silent:
        print("\nМатематическое уравнение Log-модели:")
        print(formula)
    return formula 

def print_poly_formula(
    poly_transformer: Any, 
    model: Any, 
    degree: int, 
    feature_names: List[str], 
    silent: bool = False
    ) -> str:
    """Генерирует уравнение полиномиальной регрессии с учётом степенных и перекрёстных признаков.

    Args:
        poly_transformer: Объект PolynomialFeatures.
        model: Обученная модель (sklearn) с `intercept_` и `coef_`.
        degree: Степень полинома.
        feature_names: Базовые имена признаков.
        silent: Флаг подавления вывода.

    Returns:
        Строка с уравнением (обрезается до 1000 символов для читаемости).
    """
    poly_features_names = poly_transformer.get_feature_names_out(feature_names)
    intercept_coef = model.intercept_
    coefs = model.coef_
    
    formula_parts = [f"{intercept_coef:,.2f}"]
    for c, f in zip(coefs, poly_features_names):
        sign = "+" if c >= 0 else "-"
        clean_f = f.replace(" ", "_")
        formula_parts.append(f" {sign} ({abs(c):,.4f} * {clean_f})")
        
    full_poly_formula = "Цена = " + "".join(formula_parts)
    final_formula = full_poly_formula[:1000] + "..." if len(full_poly_formula) > 1000 else full_poly_formula
    
    if not silent:
        print(f"\nМатематическое уравнение Polynomial Reg{degree}:")
        print(final_formula)
    return final_formula

# ==============================================================================
# 4. ОБУЧЕНИЕ МОДЕЛЕЙ (ядро анализа)
# ==============================================================================

def run_linear_regression(
    data: Dict[str, Any], 
    cov_type: str = 'nonrobust', 
    silent: bool = False
    ) -> Dict[str, Any]:
    """Обучает множественную линейную регрессию (OLS) и собирает результаты.

    Оценивает параметры модели методом наименьших квадратов, формирует матрицу 
    признаков с константой, рассчитывает метрики качества, таблицу дисперсионного 
    анализа и эконометрическую интерпретацию коэффициентов.

    Args:
        data: Словарь-контекст с ключами 'x_train', 'x_test', 'y_train', 'y_test'.
        cov_type: Тип ковариационной матрицы для расчёта стандартных ошибок 
                  ('nonrobust', 'HC3' и др.).
        silent: Флаг подавления детального вывода в консоль.

    Returns:
        Словарь с объектом модели, прогнозами, метриками, таблицей ANOVA, 
        интерпретацией, формулой и вспомогательными матрицами.
    """
    x_train, x_test = data['x_train'], data['x_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    x_train_sm = sm.add_constant(x_train, has_constant='add')
    x_test_sm = sm.add_constant(x_test, has_constant='add')
    model = sm.OLS(y_train, x_train_sm).fit(cov_type=cov_type)
    
    anova_table, sse_sum, mse_anova = calculate_anova(
        y_train, model.fittedvalues, x_train.shape[0], x_train.shape[1]
    )
    interpretation = get_model_interpretation(model, x_train, y_train)
    preds = model.predict(x_test_sm)
    metrics = calculate_regression_metrics(y_test, preds, "Linear Reg", model_obj=model, silent=silent)

    if not silent:
        print("\n" + "="*30 + " ANOVA TABLE " + "="*30)
        print(anova_table.to_string(index=False))
        print("\n" + "="*30 + " MODEL SUMMARY " + "="*30)
        print(model.summary()) 
        print("\nЗначимость признаков и эластичность:")
        print(interpretation.sort_values(by='Коэффициент', ascending=False).to_string(index=False))

    formula = print_linear_model_equation(model, x_train.columns, silent=silent)

    return {
        'model_object': model, 
        'predictions': preds,
        'metrics': metrics, 
        'x_train_sm': x_train_sm, 
        'n_features': x_train.shape[1], 
        'cov_type': cov_type,
        'anova_table': anova_table, 
        'sse_sum': sse_sum, 
        'residual_variance': mse_anova, 
        'residuals': model.resid,
        'Predictions': preds, 
        'Interpretation': interpretation, 
        'formula': formula,
        **metrics
    }

def run_standardized_regression(
    data: Dict[str, Any], 
    silent: bool = False
    ) -> Dict[str, Any]:
    """Обучает регрессию на стандартизированных признаках для сравнения весов.

    Приводит все предикторы к единому масштабу (нулевое среднее, единичная дисперсия),
    что позволяет напрямую сравнивать абсолютные величины коэффициентов как меру 
    относительной важности признаков.

    Args:
        data: Словарь-контекст с ключами 'x_train', 'x_test', 'y_train', 'y_test'.
        silent: Флаг подавления вывода в консоль.

    Returns:
        Словарь результатов со скалером, стандартизированными метриками и объектом модели.
    """
    x_train, x_test = data['x_train'], data['x_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    scaler_obj = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler_obj.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test_scaled = pd.DataFrame(scaler_obj.transform(x_test), columns=x_train.columns, index=x_test.index)
    
    x_train_sm = sm.add_constant(x_train_scaled)
    x_test_sm = sm.add_constant(x_test_scaled)
    model = sm.OLS(y_train, x_train_sm).fit()

    anova_table, sse_sum, mse_anova = calculate_anova(y_train, model.fittedvalues, x_train.shape[0], x_train.shape[1])
    interpretation = get_model_interpretation(model, x_train_scaled, y_train) 
    preds = model.predict(x_test_sm)
    metrics = calculate_regression_metrics(y_test, preds, "Standardized Reg", model_obj=model, silent=silent)

    if not silent:
        print("\n" + "="*20 + " ANOVA (STANDARDIZED) " + "="*20)
        print(anova_table.to_string(index=False))
        print("\n" + "="*20 + " MODEL SUMMARY " + "="*20)
        print(model.summary())
        print("\nРейтинг значимости (Станд. коэффициенты):")
        importance_df = interpretation[['Признак', 'Коэффициент', 'P-value']].copy()
        importance_df.rename(columns={'Коэффициент': 'Станд. Коэф (Вес)'}, inplace=True)
        print(importance_df.sort_values(by='Станд. Коэф (Вес)', key=abs, ascending=False).to_string(index=False))

    return {
        'model_object': model, 
        'predictions': preds, 
        'metrics': metrics, 
        'x_train_sm': x_train_sm, 
        'n_features': x_train.shape[1], 
        'cov_type': 'nonrobust',
        'Predictions': preds, 
        'Interpretation': interpretation, 
        'scaler': scaler_obj, 
        'anova_table': anova_table, 
        'sse_sum': sse_sum, 
        **metrics
    }

def run_logarithmic_regression(
    data: Dict[str, Any], 
    silent: bool = False
    ) -> Dict[str, Any]:
    """Обучает линейную регрессию по логарифмированной целевой переменной.

    Применяет преобразование log1p(y) для стабилизации дисперсии и работы с 
    правосторонними асимметричными распределениями. Коэффициенты интерпретируются 
    как полуэластичности, а прогнозы обратно преобразуются через expm1.

    Args:
        data: Словарь-контекст с ключами 'x_train', 'x_test', 'y_train', 'y_test'.
        silent: Флаг подавления вывода в консоль.

    Returns:
        Словарь результатов с логарифмическими метриками, формулой и объектом модели.
    """
    x_train, x_test = data['x_train'], data['x_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    y_train_log, y_test_log = np.log1p(y_train), np.log1p(y_test)
    x_train_const, x_test_const = sm.add_constant(x_train), sm.add_constant(x_test)
    
    model = sm.OLS(y_train_log, x_train_const).fit()
    preds_rub = np.expm1(model.predict(x_test_const))
    
    metrics = calculate_regression_metrics(
        y_test, preds_rub, "Logarithmic Reg", 
        model_obj_log=model, y_log_true=y_test_log, 
        y_log_pred=model.predict(x_test_const), 
        n_features=x_train.shape[1], silent=silent
    )
    interpretation = get_model_interpretation(model, x_train, y_train, is_log_model=True)

    if not silent:
        print("\n" + "="*30 + " MODEL SUMMARY (LOG) " + "="*30)
        print(model.summary())
        print("\nЗначимость признаков в логарифмической регрессии:")
        print(interpretation.sort_values(by='Влияние на цену (%)', ascending=False).to_string(index=False))

    formula = print_log_model_equation(model, x_train.columns, silent=silent)
    
    return {
        'model_object': model, 
        'predictions': preds_rub, 
        'metrics': metrics, 
        'x_train_sm': x_train_const, 
        'n_features': x_train.shape[1], 
        'cov_type': 'nonrobust',
        'Predictions': preds_rub, 
        'Interpretation': interpretation,
        'formula': formula,
        **metrics
    }

def run_polynomial_regression(
    data: Dict[str, Any], 
    degree: int, 
    silent: bool = False
    ) -> Dict[str, Any]:
    """Обучает полиномиальную регрессию с заданной степенью разложения признаков.

    Генерирует степенные и перекрёстные комбинации исходных признаков до указанной 
    степени, после чего обучает линейную модель на расширенном пространстве признаков.

    Args:
        data: Словарь-контекст с ключами 'x_train', 'x_test', 'y_train', 'y_test'.
        degree: Степень полиномиального разложения (обычно 2 или 3).
        silent: Флаг подавления вывода в консоль.

    Returns:
        Словарь результатов с полиномиальным трансформером, метриками и интерпретацией.
    """
    x_train, x_test = data['x_train'], data['x_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    pl_features = PolynomialFeatures(degree=degree, include_bias=False)
    x_train_poly = pl_features.fit_transform(x_train)
    x_test_poly = pl_features.transform(x_test)
    
    model = LinearRegression().fit(x_train_poly, y_train)
    preds = model.predict(x_test_poly)
    
    model_name = f"Polynomial Reg (deg={degree})"
    metrics = calculate_regression_metrics(y_test, preds, model_name, n_features=x_train_poly.shape[1], silent=silent)
    
    anova_table, _, _ = calculate_anova(y_train, model.predict(x_train_poly), x_train_poly.shape[0], x_train_poly.shape[1])
    base_feats = list(x_train.columns)

    formula = print_poly_formula(pl_features, model, degree, base_feats, silent=silent)
    if not silent:
        print(f"Топ-5 полиномиальных признаков:\n{pd.DataFrame({'Признак': pl_features.get_feature_names_out(base_feats), 'Коэф': model.coef_}).sort_values('Коэф', key=abs, ascending=False).head().to_string(index=False)}")

    return {
        'model_object': model, 
        'predictions': preds, 
        'Interpretation': pd.DataFrame({
            'Признак': pl_features.get_feature_names_out(base_feats), 
            'Коэффициент': model.coef_, 
            'P-value': [0.000]*len(model.coef_)
        }).sort_values('Коэффициент', key=abs, ascending=False),
        'metrics': metrics, 
        'x_train_sm': x_train_poly, 
        'n_features': x_train_poly.shape[1],
        'cov_type': f'Polynomial (deg {degree})', 
        'poly_transformer': pl_features,
        'anova_table': anova_table, 
        'feature_names': list(x_train.columns),
        'formula': formula,
        'Predictions': preds, 
        **metrics
    }

def run_regularized_poly_regression(
    data: Dict[str, Any], 
    degree: int, 
    reg_type: str = 'ridge', 
    alpha: Optional[float] = None, 
    silent: bool = False
) -> Dict[str, Any]:
    """Обучает регуляризованную полиномиальную регрессию (Ridge или Lasso).

    Обязательное масштабирование признаков предотвращает некорректное штрафование 
    коэффициентов регуляризационным членом. Ridge подавляет все веса пропорционально, 
    Lasso обнуляет наименее значимые, выполняя автоматический отбор признаков.

    Args:
        data: Словарь-контекст с ключами 'x_train', 'x_test', 'y_train', 'y_test'.
        degree: Степень полиномиального разложения.
        reg_type: Тип регуляризации ('ridge' или 'lasso').
        alpha: Коэффициент регуляризации. По умолчанию 1.0 для Ridge, 0.1 для Lasso.
        silent: Флаг подавления вывода в консоль.

    Returns:
        Словарь результатов со скалером, трансформером, статистикой ненулевых 
        коэффициентов и метриками.
    """
    x_train, x_test = data['x_train'], data['x_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_poly)
    x_test_scaled = scaler.transform(x_test_poly)

    if reg_type.lower() == 'ridge':
        alpha = alpha if alpha is not None else 1.0
        model = Ridge(alpha=alpha, random_state=CONFIG['RANDOM_STATE'])
        model_name = f"Ridge Poly (deg={degree}, α={alpha})"
    elif reg_type.lower() == 'lasso':
        alpha = alpha if alpha is not None else 0.1
        model = Lasso(alpha=alpha, random_state=CONFIG['RANDOM_STATE'], max_iter=10000)
        model_name = f"Lasso Poly (deg={degree}, α={alpha})"
    else:
        raise ValueError("reg_type должен быть 'ridge' или 'lasso'")

    model.fit(x_train_scaled, y_train)
    preds = model.predict(x_test_scaled)

    metrics = calculate_regression_metrics(y_test, preds, model_name, n_features=x_train_poly.shape[1], silent=silent)

    if not silent:
        nnz = int(np.sum(model.coef_ != 0))
        print(f"\n{model_name} обучена успешно.")
        print(f"Ненулевых коэффициентов: {nnz} / {len(model.coef_)}")
        print("Признаки отмасштабированы. Прямая интерпретация весов требует обратного преобразования.")

    importance_df = pd.DataFrame({
        'Признак': poly.get_feature_names_out(x_train.columns),
        'Коэффициент': model.coef_,
        'Бета-коэффициент': model.coef_,
        'P-value': [0.000] * len(model.coef_),
        'Эластичность (%)': [0.000] * len(model.coef_)
    }).sort_values(by='Коэффициент', key=abs, ascending=False)

    formula = print_poly_formula(poly, model, degree, x_train.columns.tolist(), silent=silent)
    
    return {
        'model_object': model, 
        'predictions': preds, 
        'Interpretation': importance_df,
        'metrics': metrics, 
        'x_train_sm': x_train_scaled, 
        'n_features': x_train_poly.shape[1], 
        'cov_type': f'Regularized ({reg_type})',
        'scaler': scaler, 
        'poly_transformer': poly,
        'feature_names': x_train.columns.tolist(),
        'formula': formula, 
        'Predictions': preds, 
        'nnz': int(np.sum(model.coef_ != 0)),
        'nnz_msg': f"Ненулевых коэффициентов: {int(np.sum(model.coef_ != 0))} / {len(model.coef_)}\nПризнаки отмасштабированы...",
        **metrics
    }

def run_random_forest(
    data: Dict[str, Any], 
    silent: bool = False
    ) -> Dict[str, Any]:
    """Обучает ансамблевую модель Random Forest Regressor методом бутстрэпа.

    Модель строит ансамбль из 100 решающих деревьев, каждое на случайной 
    подвыборке данных и признаков. Снижает дисперсию прогноза за счёт усреднения,
    устойчива к переобучению и не требует масштабирования признаков.

    Args:
        data: Контекст пайплайна с ключами 'x_train', 'x_test', 'y_train', 'y_test'.
        silent: Флаг подавления вывода в консоль.

    Returns:
        Словарь с объектом модели, предсказаниями, метриками качества и
        DataFrame ранжирования важности признаков.
    """
    x_train, x_test = data['x_train'], data['x_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    model = RandomForestRegressor(
        n_estimators=100, 
        random_state=CONFIG['RANDOM_STATE'],
        n_jobs=-1
    ).fit(x_train, y_train)
    
    preds = model.predict(x_test)
    metrics = calculate_regression_metrics(y_test, preds, "Random Forest", silent=silent)
    
    importance_df = pd.DataFrame({
        'Признак': x_train.columns,
        'Важность': model.feature_importances_,
        'P-value': 0.000
    }).sort_values(by='Важность', ascending=False)

    if not silent:
        print("\n--- Топ-5 важных признаков (Random Forest) ---")
        print(importance_df.head(5).to_string(index=False))

    return {
        'model_object': model, 
        'predictions': preds, 
        'Interpretation': importance_df,
        'metrics': metrics, 
        'x_train_sm': x_train,
        'n_features': x_train.shape[1], 
        'cov_type': 'Ensemble (Bagging)', 
        'Predictions': preds, 
        **metrics
    }

def run_catboost(
    data: Dict[str, Any], 
    silent: bool = False
    ) -> Dict[str, Any]:
    """Обучает градиентный бустинг CatBoost Regressor.

    Использует симметричные деревья и обработку категориальных признаков 
    без предварительного кодирования. Минимизирует функцию потерь последовательно,
    исправляя ошибки предыдущих деревьев, что обеспечивает высокую точность.

    Args:
        data: Контекст пайплайна с ключами 'x_train', 'x_test', 'y_train', 'y_test'.
        silent: Флаг подавления вывода в консоль.

    Returns:
        Словарь с объектом модели, предсказаниями, метриками качества и
        DataFrame ранжирования важности признаков.
    """
    x_train, x_test = data['x_train'], data['x_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    model = CatBoostRegressor(
        iterations=100, 
        depth=7, 
        learning_rate=0.1, 
        silent=True, 
        allow_writing_files=False,
        random_seed=CONFIG['RANDOM_STATE']
    ).fit(x_train, y_train)
    
    preds = model.predict(x_test)
    metrics = calculate_regression_metrics(y_test, preds, "CatBoost", silent=silent)
    
    importance_df = pd.DataFrame({
        'Признак': x_train.columns,
        'Важность (%)': model.get_feature_importance(),
        'P-value': 0.000
    }).sort_values(by='Важность (%)', ascending=False)
    
    if not silent:
        print("\n--- Важность признаков CatBoost ---")
        print(importance_df.head(10).to_string(index=False))

    return {
        'model_object': model, 
        'predictions': preds, 
        'Interpretation': importance_df,
        'metrics': metrics, 
        'x_train_sm': x_train,
        'n_features': x_train.shape[1], 
        'cov_type': 'Gradient Boosting', 
        **metrics
    }

def build_baseline_models(
    data: Dict[str, Any], 
    silent: bool = False
    ) -> Dict[str, Any]:
    """Оркестрирует обучение набора базовых моделей для сравнительного анализа.

    Последовательно запускает линейные, стандартизированные, логарифмические,
    полиномиальные, регуляризованные и ансамблевые алгоритмы. Результаты 
    агрегируются в единый словарь для дальнейшей диагностики и отчётности.

    Args:
        data: Контекст пайплайна с ключами 'x_train', 'x_test', 'y_train', 'y_test'.
        silent: Флаг подавления детального вывода для каждой подмодели.

    Returns:
        Словарь, где ключи -- идентификаторы моделей, значения -- словари 
        с результатами обучения (объекты, метрики, интерпретации).
    """
    models_results = {}
    
    print("\n" + "="*60)
    print("ЭТАП 4: ПОСТРОЕНИЕ БАЗОВЫХ МОДЕЛЕЙ")
    print("="*60)

    print("\n=== 1. Множественная линейная регрессия ===")
    models_results['linear'] = run_linear_regression(data, cov_type='nonrobust', silent=silent)
    
    print("\n=== 2. Стандартизированная линейная регрессия ===")
    models_results['standardized'] = run_standardized_regression(data, silent=silent)
    
    print("\n=== 3. Линейная регрессия по логарифмам ===")
    models_results['log'] = run_logarithmic_regression(data, silent=silent)
    
    print("\n=== 4. Полиномиальная регрессия (степень 2) ===")
    models_results['poly2'] = run_polynomial_regression(data, degree=2, silent=silent)
    
    print("\n=== 5. Полиномиальная регрессия (степень 3) ===")
    models_results['poly3'] = run_polynomial_regression(data, degree=3, silent=silent)

    print("\n=== 6. Полиномиальная регрессия с ридж (степень 2) ===")
    models_results['ridge_poly2'] = run_regularized_poly_regression(data, degree=2, reg_type='ridge', silent=silent)

    print("\n=== 7. Полиномиальная регрессия с лассо (степень 2) ===")
    models_results['lasso_poly2'] = run_regularized_poly_regression(data, degree=2, reg_type='lasso', silent=silent)

    print("\n=== 8. Нелинейная регрессия (Случайный лес) ===")
    models_results['random_forest'] = run_random_forest(data, silent=silent)
    
    print("\n=== 9. Нелинейная регрессия CatBoost ===")
    models_results['catboost'] = run_catboost(data, silent=silent)

    print("\n" + "-"*60)
    print(f"Все модели ({len(models_results)}) успешно созданы")
    
    return models_results

# ==============================================================================
# 5. СТАТИСТИЧЕСКАЯ ДИАГНОСТИКА (тесты по отдельности)
# ==============================================================================

def run_fisher_test(
    data: Dict[str, Any], 
    alpha: float = CONFIG["ALPHA"], 
    silent: bool = False
    ) -> None:
    """Проверяет общую статистическую значимость регрессионного уравнения с помощью F-теста.

    Сравнивает расчетное значение F-статистики с критическим значением распределения Фишера
    при заданном уровне значимости. Отвергает нулевую гипотезу о равенстве всех коэффициентов нулю.

    Args:
        data: Словарь-контекст с ключом 'model_object' (обученная модель statsmodels).
        alpha: Уровень значимости для расчета критического значения.
        silent: Флаг подавления вывода в консоль.
    """
    model = data['model_object']
    dfn, dfd = model.df_model, model.df_resid
    f_crit = stats.f.ppf(1 - alpha, dfn, dfd)
    p_val = model.f_pvalue

    if not silent:
        print("\n--- F-ТЕСТ (Значимость уравнения в целом) ---")
        print(f"F-расчетное: {model.fvalue:.2f} | F-критическое: {f_crit:.2f}")
        print(f"p-value: {p_val:.4e} (Порог: {alpha})")
        status = "ЗНАЧИМА" if p_val < alpha else "НЕ ЗНАЧИМА"
        print(f"Вердикт: Модель {status}")

def run_t_test(
    data: Dict[str, Any], 
    alpha: float = CONFIG["ALPHA"], 
    silent: bool = False
    ) -> pd.DataFrame:
    """Проверяет статистическую значимость отдельных коэффициентов регрессии.

    Рассчитывает t-статистику для каждого параметра и сравнивает с критическим значением
    распределения Стьюдента. Формирует DataFrame с вердиктами о значимости признаков.

    Args:
        data: Словарь-контекст с ключом 'model_object'.
        alpha: Уровень значимости.
        silent: Флаг подавления вывода.

    Returns:
        DataFrame с колонками 't-стат', 'P-value', 'Вердикт', отсортированный по 
        абсолютной величине t-статистики.
    """
    model = data['model_object']
    a = alpha if alpha < 0.5 else (1 - alpha)
    t_crit = stats.t.ppf(1 - a/2, model.df_resid)

    t_diag = pd.DataFrame({
        't-стат': model.tvalues,
        'P-value': model.pvalues
    })
    t_diag['Вердикт'] = t_diag['t-стат'].abs().apply(
        lambda x: f"ЗНАЧИМ (|{x:.2f}| > {t_crit:.2f})" if x > t_crit 
        else f"Шум (|{x:.2f}| < {t_crit:.2f})"
    )

    if not silent:
        print("\n--- t-ТЕСТ (Значимость отдельных коэффициентов) ---")
        print(f"Критическое значение t_crit: {t_crit:.3f} (df={model.df_resid:.0f}, alpha={a})")
        print(t_diag[['t-стат', 'P-value', 'Вердикт']].sort_values('t-стат', key=abs, ascending=False).to_string())

    return t_diag

def run_t_intervals(
    data: Dict[str, Any], 
    alpha: float = CONFIG["ALPHA"], 
    silent: bool = False
    ) -> pd.DataFrame:
    """Рассчитывает доверительные интервалы для коэффициентов регрессии.

    Оценивает диапазон, в котором с заданной вероятностью находится истинное значение
    параметра. Если интервал содержит ноль, коэффициент признается статистически незначимым.

    Args:
        data: Словарь-контекст с ключом 'model_object'.
        alpha: Уровень значимости.
        silent: Флаг подавления вывода.

    Returns:
        DataFrame с коэффициентами, границами интервалов и статусом надежности.
    """
    model = data['model_object']
    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100
    conf_int = model.conf_int(alpha)

    ci_diag = pd.DataFrame({
        'Коэффициент': model.params,
        f'Нижняя граница ({lower_pct:.1f}%)': conf_int[0],
        f'Верхняя граница ({upper_pct:.1f}%)': conf_int[1]
    })
    ci_diag['Надежен?'] = ci_diag.apply(
        lambda row: "ДА" if (row.iloc[1] * row.iloc[2] > 0) else f"НЕТ (содержит 0 при α={alpha})",
        axis=1
    )

    if not silent:
        confidence_level = int((1 - alpha) * 100)
        print(f"\n[!] Анализ {confidence_level}% доверительных интервалов (α={alpha}):")
        print("-" * 80)
        print(ci_diag.to_string())
        print("-" * 80)

    return ci_diag

def run_point_prediction(
    data: Dict[str, Any], 
    x_test: pd.DataFrame, 
    apartment_index: int = CONFIG["APARTMENT_INDEX"], 
    silent: bool = False
    ) -> Dict[str, Any]:
    """Формирует точечный прогноз и доверительные интервалы для конкретного наблюдения.

    Рассчитывает ожидаемое значение целевой переменной, а также интервалы для среднего
    (ожидаемая цена на рынке) и индивидуального (вилка цены для конкретного объекта) значения.

    Args:
        data: Словарь-контекст с ключом 'model_object'.
        x_test: Тестовая выборка признаков.
        apartment_index: Индекс объекта в тестовой выборке.
        silent: Флаг подавления вывода.

    Returns:
        Словарь с точечным прогнозом ('prediction'), границами ДИ для среднего ('ci_mean') 
        и индивидуального ('ci_individual') значений.
    """
    x_test_row = x_test.iloc[[apartment_index]]
    model = data['model_object']
    x_processed = x_test_row.copy()
    x0 = sm.add_constant(x_processed, has_constant='add')
    x0 = x0[model.model.exog_names]

    prediction_obj = model.get_prediction(x0)
    y_pred = prediction_obj.predicted_mean
    ci_mean = prediction_obj.conf_int(obs=False, alpha=CONFIG["ALPHA"])
    ci_ind = prediction_obj.conf_int(obs=True, alpha=CONFIG["ALPHA"])
    ci_ind[0][0] = max(0, ci_ind[0][0])

    if not silent:
        print("\n" + "-"*10 + f" ХАРАКТЕРИСТИКИ ОБЪЕКТА (Тестовый индекс: {apartment_index}) " + "-"*10)
        print(x_test_row.to_string(index=False))
        print("\n" + "="*20 + " РЕЗУЛЬТАТЫ ПРОГНОЗА " + "="*20)
        print(f"Точечный прогноз цены: {y_pred[0]:,.0f} руб.")
        print(f"\n{int((1-CONFIG['ALPHA'])*100)}% ДИ для СРЕДНЕГО значения (рыночная норма):")
        print(f"  [{ci_mean[0][0]:,.0f} - {ci_mean[0][1]:,.0f}] руб.")
        print(f"\n{int((1-CONFIG['ALPHA'])*100)}% ДИ для ИНДИВИДУАЛЬНОГО значения (вилка для объекта):")
        print(f"  [{ci_ind[0][0]:,.0f} - {ci_ind[0][1]:,.0f}] руб.")

    return {
        'prediction': y_pred[0],
        'ci_mean': ci_mean[0],
        'ci_individual': ci_ind[0]
    }


def check_residuals_randomness(
    data: Dict[str, Any], 
    alpha: float = CONFIG["ALPHA"], 
    silent: bool = False
    ) -> pd.DataFrame:
    """Проверяет остатки на наличие автокорреляции с помощью теста Льюнга-Бокса.

    Нулевая гипотеза предполагает, что остатки представляют собой белый шум (отсутствует
    систематическая зависимость между ошибками на разных лагах).

    Args:
        data: Контекст с 'model_object'.
        alpha: Уровень значимости.
        silent: Подавление вывода.

    Returns:
        DataFrame с результатами теста на различных лагах.
    """
    resid = data['model_object'].resid
    lags = [10, 20, 100]
    lb_test = acorr_ljungbox(resid, lags=lags, return_df=True)
    min_p_val = lb_test['lb_pvalue'].min()

    if not silent:
        print("\n--- Тест Льюнга-Бокса (Проверка на белый шум) ---")
        print(lb_test.to_string())
        if min_p_val > alpha:
            print("\n[OK] Вердикт: Остатки — 'белый шум' (p-min={min_p_val:.4f}).")
            print("     Модель полностью извлекла информацию из признаков.")
        else:
            print("\n[!] Вердикт: Обнаружена автокорреляция остатков (p-min={min_p_val:.4f}).")
            print("     Возможно, пропущен важный нелинейный фактор или есть кластеризация данных.")

    return lb_test

def check_zero_mean(
    data: Dict[str, Any], 
    silent: bool = False
    ) -> float:
    """Проверяет условие Гаусса-Маркова о нулевом математическом ожидании остатков.

    Среднее значение остатков должно быть близко к нулю. Отклонение указывает на
    систематическую ошибку модели или отсутствие константы в уравнении.

    Args:
        data: Контекст с 'model_object'.
        silent: Подавление вывода.

    Returns:
        Среднее значение остатков.
    """
    resid = data['model_object'].resid
    mean_resid = np.mean(resid)

    if not silent:
        print(f"Среднее остатков: {mean_resid:.4e}")
        if np.abs(mean_resid) < 1e-7:
            print("Результат: Условие Гаусса-Маркова выполнено (среднее равно нулю).")
        else:
            print("Результат: Условие НЕ выполнено. Возможно, модель обучена без константы.")

    return mean_resid

def run_breusch_pagan_test(
    data: Dict[str, Any], 
    alpha: float = CONFIG["ALPHA"], 
    silent: bool = False
    ) -> float:
    """Тестирует гомоскедастичность остатков (постоянство дисперсии ошибок).

    Нулевая гипотеза: дисперсия остатков постоянна. Отклонение указывает на
    гетероскедастичность, что делает обычные стандартные ошибки ненадежными.

    Args:
        data: Контекст с 'model_object' и 'x_train_sm' (матрица признаков с константой).
        alpha: Уровень значимости.
        silent: Подавление вывода.

    Returns:
        p-value теста Бреуша-Пагана.
    """
    resid = data['model_object'].resid
    X = data['x_train_sm']
    lm_stat, p_value, _, _ = het_breuschpagan(resid, X)

    if not silent:
        print("\n" + "-"*10 + " ТЕСТ БРЕУША-ПАГАНА " + "-"*10)
        print(f"Lagrange Multiplier stat: {lm_stat:.4f}")
        print(f"p-value: {p_value:.4e}")
        if p_value < alpha:
            print("РЕЗУЛЬТАТ: Обнаружена гетероскедастичность (H0 отвергнута).")
            print("ВЛИЯНИЕ: Оценки коэффициентов несмещены, но их стандартные ошибки ненадёжны.")
            print("СОВЕТ: Используйте робастные стандартные ошибки (HC3) для проверки значимости.")
        else:
            print("РЕЗУЛЬТАТ: Гетероскедастичность не обнаружена (H0 подтверждена).")

    return p_value

def run_goldfeld_quandt_test(
    data: Dict[str, Any], 
    alpha: float = CONFIG["ALPHA"], 
    silent: bool = False
    ) -> float:
    """Сравнивает дисперсии ошибок в двух частях выборки для выявления гетероскедастичности.

    Выборка сортируется по одному из признаков, делится на две части, и для них
    рассчитываются независимые регрессии. F-статистика сравнивает остатки.

    Args:
        data: Контекст с 'model_object' и 'x_train_sm'.
        alpha: Уровень значимости.
        silent: Подавление вывода.

    Returns:
        p-value теста Голдфелда-Квандта.
    """
    resid = data['model_object'].resid
    X = data['x_train_sm']
    f_stat, p_value, direction = het_goldfeldquandt(resid, X)

    if not silent:
        print("\n" + "-"*10 + " ТЕСТ ГОЛДФЕЛДА-КВАНДТА " + "-"*10)
        print(f"F-статистика: {f_stat:.4f}")
        print(f"p-value: {p_value:.4e}")
        directions = {
            'increasing': 'Рост дисперсии (увеличение разброса ошибок)',
            'decreasing': 'Убывание дисперсии (сужение разброса ошибок)',
            'two-sided': 'Двустороннее изменение дисперсии'
        }
        print(f"Направление: {directions.get(direction, direction)}")
        if p_value < alpha:
            print("Результат: Гетероскедастичность ОБНАРУЖЕНА.")
            print("Вывод: Дисперсия ошибок значимо различается между частями выборки.")
            print("Рекомендация: Проверьте структуру данных или используйте робастные ошибки (HC).")
        else:
            print("Результат: Гомоскедастичность подтверждена (H0 не отвергается).")
            print("Вывод: Дисперсия остатков стабильна на всей выборке.")

    return p_value

def run_durbin_watson_test(
    data: Dict[str, Any], 
    silent: bool = False
    ) -> float:
    """Оценивает наличие автокорреляции первого порядка в остатках.

    Статистика DW близка к 2 при отсутствии автокорреляции, к 0 при положительной
    и к 4 при отрицательной. Критична для временных рядов и упорядоченных данных.

    Args:
        data: Контекст с 'model_object'.
        silent: Подавление вывода.

    Returns:
        Значение статистики Дарбина-Уотсона.
    """
    resid = data['model_object'].resid
    dw_stat = durbin_watson(resid)

    if not silent:
        print("\n" + "-"*10 + " ТЕСТ ДАРБИНА-УОТСОНА " + "-"*10)
        print(f"Значение DW-статистики: {dw_stat:.3f}")
        if 1.5 <= dw_stat <= 2.5:
            print("Результат: Автокорреляция не обнаружена (в пределах нормы).")
            print("Вывод: Остатки распределены случайно, условие независимости соблюдено.")
        elif dw_stat < 1.5:
            print("Результат: Обнаружена ПОЛОЖИТЕЛЬНАЯ автокорреляция.")
            print("Вывод: Ошибки имеют тенденцию сохранять знак. Возможен пропущенный тренд.")
        else:
            print("Результат: Обнаружена ОТРИЦАТЕЛЬНАЯ автокорреляция.")
            print("Вывод: Ошибки слишком часто меняют знак. Возможна избыточная подгонка (overfitting).")
        if dw_stat < 1.0 or dw_stat > 3.0:
            print("Критическое отклонение! Стандартные ошибки и t-статистики могут быть сильно искажены.")

    return dw_stat

def run_shapiro_test(
    data: Dict[str, Any], 
    alpha: float = CONFIG["ALPHA"], 
    silent: bool = False
    ) -> float:
    """Проверяет нормальность распределения остатков с помощью критерия Шапиро-Уилка.

    Нулевая гипотеза: выборка извлечена из нормального распределения. Критичен для
    корректности доверительных интервалов и p-value коэффициентов в МНК.

    Args:
        data: Контекст с 'model_object'.
        alpha: Уровень значимости.
        silent: Подавление вывода.

    Returns:
        p-value теста Шапиро-Уилка.
    """
    resid = data['model_object'].resid
    n = len(resid)
    shapiro_stat, shapiro_p = stats.shapiro(resid)

    if not silent:
        print("\n" + "-"*10 + " ТЕСТ ШАПИРО-УИЛКА " + "-"*10)
        print(f"Статистика: {shapiro_stat:.4f}, p-value: {shapiro_p:.4e}")
        if shapiro_p < alpha:
            print("РЕЗУЛЬТАТ: Остатки распределены НЕ НОРМАЛЬНО.")
            if n > 300:
                print(f"Примечание: На выборке n={n} тест может быть избыточно строгим.")
                print("Рекомендуется проверить график QQ-plot: если точки ложатся на линию, отклонением можно пренебречь.")
            else:
                print("Вывод: Доверительные интервалы и p-value коэффициентов могут быть неточными.")
        else:
            print("РЕЗУЛЬТАТ: Остатки распределены НОРМАЛЬНО.")
            print("Вывод: Классические статистические тесты (t, F) максимально надежны.")

    return shapiro_p

def run_rs_test(
    data: Dict[str, Any], 
    silent: bool = False
    ) -> float:
    """Оценивает наличие выбросов через соотношение размаха и стандартного отклонения остатков.

    Эмпирический критерий, чувствительный к тяжелым хвостам распределения. Значения выше 8
    указывают на аномальные наблюдения, значительно искажающие модель.

    Args:
        data: Контекст с 'model_object'.
        silent: Подавление вывода.

    Returns:
        Значение RS-статистики.
    """
    resid = data['model_object'].resid
    n = len(resid)
    std_resid = np.std(resid)
    range_resid = np.max(resid) - np.min(resid)
    rs_stat = range_resid / std_resid

    if not silent:
        print("\n" + "-"*10 + " RS-КРИТЕРИЙ (Range/Std) " + "-"*10)
        print(f"Статистика (q): {rs_stat:.2f} (n={n})")
        if rs_stat > 8.0:
            print("РЕЗУЛЬТАТ: Обнаружены СИЛЬНЫЕ ВЫБРОСЫ (тяжелые хвосты).")
            print("Вывод: Несколько ошибок модели значительно превышают типичный разброс.")
            print("Совет: Проверьте данные на наличие аномалий или используйте Robust регрессию.")
        elif rs_stat < 3.0:
            print("РЕЗУЛЬТАТ: Слишком узкий разброс (короткие хвосты).")
        else:
            print("РЕЗУЛЬТАТ: Соотношение размаха к отклонению в пределах нормы.")
            print("Вывод: Структура хвостов распределения соответствует нормальному.")

    return rs_stat

def run_jarque_bera_test(
    data: Dict[str, Any], 
    alpha: float = CONFIG["ALPHA"], 
    silent: bool = False
    ) -> Tuple[float, float]:
    """Проверяет нормальность распределения на основе асимметрии и эксцесса остатков.

    Асимптотический тест, основанный на отклонении выборочной асимметрии и эксцесса
    от значений нормального распределения (0 и 3 соответственно).

    Args:
        data: Контекст с 'model_object'.
        alpha: Уровень значимости.
        silent: Подавление вывода.

    Returns:
        Кортеж (статистика JB, p-value).
    """
    resid = data['model_object'].resid
    jb_stat, p_value = stats.jarque_bera(resid)
    skew = stats.skew(resid)
    kurt = stats.kurtosis(resid, fisher=False)

    if not silent:
        print("\n" + "-"*10 + " ТЕСТ ЖАКА-БЕРА (JARQUE-BERA) " + "-"*10)
        print(f"Статистика JB: {jb_stat:.2f}")
        print(f"p-value: {p_value:.4e}")
        print(f"Асимметрия (Skew): {skew:.3f} (0 - симметрия)")
        print(f"Эксцесс (Kurt): {kurt:.3f} (3 - нормальная острота)")
        if p_value < alpha:
            print("Результат: Распределение НЕ нормальное.")
            reasons = []
            if abs(skew) > 0.5: reasons.append("сильная асимметрия")
            if abs(kurt - 3) > 1: reasons.append("нетипичная острота пика/хвостов")
            if reasons:
                print(f"Причина: {', '.join(reasons)}.")
        else:
            print("Результат: Распределение близко к нормальному.")

    return jb_stat, p_value

def run_chow_test_logic(
    x_train: pd.DataFrame, 
    y_train: pd.Series, 
    mask: pd.Series, 
    test_name: str, 
    alpha: float = CONFIG["ALPHA"], 
    silent: bool = False
    ) -> Optional[Tuple[float, float]]:
    """Проверяет структурную стабильность модели при разбиении выборки на подгруппы.

    Сравнивает сумму квадратов ошибок единой модели с суммой ошибок раздельных моделей
    для каждой подгруппы. Значимое различие указывает на смену генерирующего процесса.

    Args:
        x_train: Матрица признаков обучающей выборки.
        y_train: Вектор целевой переменной.
        mask: Булева маска для разделения выборки на две части.
        test_name: Наименование теста для вывода в лог.
        alpha: Уровень значимости.
        silent: Подавление вывода.

    Returns:
        Кортеж (F-статистика, p-value) или None при недостатке данных.
    """
    res_total = run_linear_regression({'x_train': x_train, 'x_test': x_train, 'y_train': y_train, 'y_test': y_train}, silent=True)
    sse_total = res_total['sse_sum']

    x1, y1 = x_train[mask], y_train[mask]
    x2, y2 = x_train[~mask], y_train[~mask]

    if not silent:
        print("\n" + "!"*20 + f" ТЕСТ ЧОУ: {test_name} " + "!"*20)
        print(f"Размер группы 1: {len(y1)} | Размер группы 2: {len(y2)}")

    k_params = x_train.shape[1] + 1
    if len(y1) < k_params or len(y2) < k_params:
        if not silent:
            print(f"Ошибка: Одна из выборок слишком мала (нужно минимум {k_params} строк).")
        return None

    x1_filtered = x1.loc[:, x1.std() > 0]
    x2_filtered = x2.loc[:, x2.std() > 0]

    res1 = run_linear_regression({'x_train': x1_filtered, 'x_test': x1_filtered, 'y_train': y1, 'y_test': y1}, silent=True)
    res2 = run_linear_regression({'x_train': x2_filtered, 'x_test': x2_filtered, 'y_train': y2, 'y_test': y2}, silent=True)

    sse1, sse2 = res1['sse_sum'], res2['sse_sum']
    n = len(y_train)
    k = k_params

    numerator = (sse_total - (sse1 + sse2)) / k
    denominator = (sse1 + sse2) / (n - 2 * k)
    f_chow = numerator / denominator if denominator != 0 else 0

    f_crit = stats.f.ppf(1 - alpha, k, n - 2 * k)
    p_value = 1 - stats.f.cdf(f_chow, k, n - 2 * k)

    if not silent:
        print("\nРЕЗУЛЬТАТ ТЕСТА {}:".format(test_name))
        print(f"F-расчетное: {f_chow:.4f} (F-крит: {f_crit:.4f})")
        p_str = f"{p_value:.4e}" if p_value > 1e-16 else "< 1e-16"
        print(f"p-value: {p_str}")
        if p_value < alpha:
            print("ВЫВОД: Структурная стабильность НАРУШЕНА.")
            print(f"    Статистическая разница между группами подтверждена на уровне {alpha*100}%.")
            print("    Рекомендация: Постройте отдельные модели")
        else:
            print("ВЫВОД: Структура СТАБИЛЬНА.")
            print("    Различия между группами случайны. Объединение данных корректно.")

    return f_chow, p_value

def run_reset_test(
    data: Dict[str, Any], 
    alpha: float = CONFIG["ALPHA"], 
    silent: bool = False
    ) -> Any:
    """Тест Раmseя на корректность спецификации линейной модели.

    Проверяет, улучшают ли модель добавленные степени предсказанных значений (Y^2, Y^3).
    Значимый результат указывает на пропущенные нелинейные члены или взаимодействия.

    Args:
        data: Контекст с 'model_object'.
        alpha: Уровень значимости.
        silent: Подавление вывода.

    Returns:
        Результат теста (объект с атрибутами fvalue, pvalue).
    """
    model = data['model_object']
    reset = reset_ramsey(model, degree=3)

    if not silent:
        print("\n" + "="*15 + " ТЕСТ РАМСЕЯ (RESET TEST) " + "="*15)
        print(f"F-статистика: {float(reset.fvalue):.4f}")
        print(f"p-value: {reset.pvalue:.4e}")
        if reset.pvalue < alpha:
            print("Результат: Обнаружена ошибка спецификации (H0 отвергнута).")
            print("Вывод: Линейная форма модели неполна. Возможны:")
            print("      - Пропущенные квадратичные или кубические члены переменных.")
            print("      - Наличие неучтенных взаимодействий (interactions).")
            print("      - Пропуск значимых объясняющих переменных.")
        else:
            print("Результат: Спецификация модели признана адекватной.")
            print("Вывод: Линейная аппроксимация хорошо описывает зависимости в данных.")

    return reset

def run_influence_diagnostics(
    data: Dict[str, Any], 
    threshold_factor: float = CONFIG["COOKS_FACTOR"], 
    silent: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Выявляет влиятельные наблюдения и точки с высоким рычагом (leverage).

    Расстояние Кука измеряет влияние отдельного наблюдения на все коэффициенты модели.
    Leverage показывает, насколько аномальны значения признаков наблюдения.

    Args:
        data: Контекст с 'model_object'.
        threshold_factor: Множитель для порога Кука (обычно 4).
        silent: Подавление вывода.

    Returns:
        Кортеж (массив расстояний Кука, массив рычагов).
    """
    model = data['model_object']
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag

    n = len(cooks_d)
    k = len(model.params)
    cooks_threshold = threshold_factor / n
    leverage_threshold = 3 * (k / n)

    outliers_idx = np.where(cooks_d > cooks_threshold)[0]
    high_leverage_idx = np.where(leverage > leverage_threshold)[0]

    if not silent:
        print("\n" + "="*15 + " АНАЛИЗ ВЛИЯНИЯ (COOK'S & LEVERAGE) " + "="*15)
        intersection = np.intersect1d(outliers_idx, high_leverage_idx)
        print("1. Расстояние Кука (Cook's D):")
        print(f"   - Порог ({threshold_factor}/n): {cooks_threshold:.5f}")
        print(f"   - Найдено влиятельных объектов: {len(outliers_idx)}")
        print(f"   - Макс. влияние: {np.max(cooks_d):.4f}")
        print("\n2. Рычаги (Leverage):")
        print(f"   - Порог (3*k/n): {leverage_threshold:.5f}")
        print(f"   - Аномальных X-характеристик: {len(high_leverage_idx)}")
        print(f"   - Макс. рычаг: {np.max(leverage):.4f}")
        if len(intersection) > 0:
            print("\n[!!!] КРИТИЧЕСКИЕ ТОЧКИ: {} объектов одновременно имеют".format(len(intersection)))
            print("      аномальный X и сильно искажают модель. Индексы: {}".format(intersection[:10]))
        elif len(outliers_idx) > 0:
            print("\n[!] СОВЕТ: Проверьте индексы {}. Они могут быть выбросами.".format(outliers_idx[:5]))

    return cooks_d, leverage

def check_vif(
    x_train: pd.DataFrame, 
    silent: bool = False
    ) -> pd.DataFrame:
    """Рассчитывает фактор инфляции дисперсии (VIF) для выявления мультиколлинеарности.

    VIF > 10 указывает на сильную линейную зависимость признака от остальных,
    что делает оценки коэффициентов нестабильными и трудноинтерпретируемыми.

    Args:
        x_train: Матрица признаков.
        silent: Подавление вывода.

    Returns:
        DataFrame с признаками и их значениями VIF.
    """
    X_with_const = sm.add_constant(x_train, has_constant='add')
    vif_data = pd.DataFrame()
    vif_data["Признак"] = X_with_const.columns
    vif_values = []
    for i in range(X_with_const.shape[1]):
        try:
            vif_values.append(variance_inflation_factor(X_with_const.values, i))
        except Exception:
            vif_values.append(np.nan)
    vif_data["VIF"] = vif_values

    vif_results = vif_data[vif_data["Признак"] != 'const'].sort_values(by="VIF", ascending=False)

    if not silent:
        print("\n" + "="*15 + " ПРОВЕРКА МУЛЬТИКОЛЛИНЕАРНОСТИ (VIF) " + "="*15)
        print(vif_results.assign(VIF=lambda x: x.VIF.round(2)).to_string(index=False))
        high_vif = vif_results[vif_results["VIF"] > CONFIG["VIF_THRESHOLD"]]
        if not high_vif.empty:
            if np.isinf(high_vif["VIF"]).any():
                inf_features = high_vif[np.isinf(high_vif["VIF"])]["Признак"].tolist()
                print("\nКРИТИЧНО: Признаки {} имеют идеальную мультиколлинеарность (Inf)!".format(inf_features))
            print("\nВНИМАНИЕ: Найдено {} признаков с VIF > {}.".format(len(high_vif), CONFIG['VIF_THRESHOLD']))
            print("Рекомендация: Рассмотрите удаление признака с наибольшим VIF или применение PCA.")
        else:
            print("\n[OK] Все значения VIF в норме (ниже {}).".format(CONFIG['VIF_THRESHOLD']))

    return vif_results

def check_condition_number(
    x_train: pd.DataFrame, 
    silent: bool = False
    ) -> float:
    """Вычисляет число обусловленности матрицы признаков для оценки устойчивости решения.

    Число обусловленности > 30 указывает на умеренную, > 1000 на критическую
    мультиколлинеарность. Высокие значения приводят к большим ошибкам округления.

    Args:
        x_train: Матрица признаков.
        silent: Подавление вывода.

    Returns:
        Значение числа обусловленности.
    """
    X = sm.add_constant(x_train)
    cond_num = np.linalg.cond(X.values)

    if not silent:
        print("\n" + "="*15 + " CONDITION NUMBER (CN) " + "="*15)
        print("Значение: {:.2f}".format(cond_num))
        if cond_num > 1000:
            print("Результат: КРИТИЧЕСКАЯ мультиколлинеарность.")
            print("Вывод: Матрица почти вырождена. Оценкам коэффициентов нельзя доверять.")
        elif cond_num > 30:
            print("Результат: Умеренная мультиколлинеарность.")
            print("Совет: Проверьте VIF или попробуйте стандартизировать признаки (StandardScaler).")
        else:
            print("Результат: Матрица стабильна. Мультиколлинеарность не влияет на решение.")

    return cond_num

def run_dfbetas_diagnostics(
    data: Dict[str, Any], 
    silent: bool = False
    ) -> np.ndarray:
    """Оценивает влияние каждого наблюдения на конкретные коэффициенты регрессии.

    DFBetas показывает, на сколько стандартных отклонений изменится коэффициент,
    если исключить конкретное наблюдение из выборки.

    Args:
        data: Контекст с 'model_object'.
        silent: Подавление вывода.

    Returns:
        Матрица значений DFBetas (наблюдения x признаки).
    """
    model = data['model_object']
    influence = model.get_influence()
    dfbs = influence.dfbetas
    feature_names = model.params.index
    n = len(dfbs)
    threshold = 2 / np.sqrt(n)

    if not silent:
        print("\n" + "="*15 + " АНАЛИЗ DFBETAS (Влияние на конкретные веса) " + "="*15)
        print("Порог (2/sqrt(n)): {:.4f}".format(threshold))
        found_any = False
        for i, name in enumerate(feature_names):
            impact_abs = np.abs(dfbs[:, i])
            impactful_points = np.where(impact_abs > threshold)[0]
            if len(impactful_points) > 0:
                found_any = True
                max_idx = impactful_points[np.argmax(impact_abs[impactful_points])]
                max_val = impact_abs[max_idx]
                print("\n[!] Признак '{}':".format(name))
                print("    - Проблемных точек: {} ({:.1f}%)".format(len(impactful_points), (len(impactful_points)/n)*100))
                print("    - Макс. влияние: {:.4f} (индекс {})".format(max_val, max_idx))
        if not found_any:
            print("Результат: Влиятельных наблюдений для весов признаков не обнаружено.")

    return dfbs

# ==============================================================================
# 6. СБОР ДИАГНОСТИКИ И ФОРМИРОВАНИЕ АНАЛИТИЧЕСКИХ БЛОКОВ
# ==============================================================================

def collect_diagnostic_data(
    data: Dict[str, Any], 
    alpha: float = CONFIG["ALPHA"]
    ) -> Dict[str, Any]:
    """Выполняет комплексную эконометрическую диагностику модели и сохраняет результаты.

    Последовательно рассчитывает критерии значимости, мультиколлинеарности, 
    гомоскедастичности, нормальности остатков, структурной стабильности (Chow) 
    и влияния отдельных наблюдений. Все вычисления кэшируются в единый словарь.

    Args:
        data: Контекст пайплайна, содержащий ключи:
              'model_result' (результат обучения), 'x_train', 'y_train', 'x_test'.
        alpha: Уровень статистической значимости для принятия решений.

    Returns:
        Словарь с результатами всех проведённых тестов и рассчитанными метриками.
    """
    model_result = data['model_result']
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']

    model = model_result['model_object']
    resid = model.resid
    X_const = sm.add_constant(x_train, has_constant='add')
    diag = {}

    # 1. Значимость модели и коэффициентов
    dfn, dfd = model.df_model, model.df_resid
    f_crit = stats.f.ppf(1 - alpha, dfn, dfd)
    diag['fisher'] = {
        'F_calc': model.fvalue, 'F_crit': f_crit, 'p_value': model.f_pvalue,
        'status': "ЗНАЧИМА" if model.f_pvalue < alpha else "НЕ ЗНАЧИМА"
    }

    t_crit = stats.t.ppf(1 - alpha / 2, model.df_resid)
    t_df = pd.DataFrame({'t-стат': model.tvalues, 'P-value': model.pvalues})
    t_df['Вердикт'] = t_df['t-стат'].abs().apply(
        lambda x: f"ЗНАЧИМ (|{x:.2f}| > {t_crit:.2f})" if x > t_crit else "Шум"
    )
    diag['t_test'] = t_df.sort_values('t-стат', key=abs, ascending=False)
    diag['t_crit'] = t_crit

    conf_int = model.conf_int(alpha)
    ci_df = pd.DataFrame({
        'Коэффициент': model.params, 'Нижняя': conf_int[0], 'Верхняя': conf_int[1]
    })
    ci_df['Надежен?'] = ci_df.apply(
        lambda r: "ДА" if r['Нижняя'] * r['Верхняя'] > 0 else f"НЕТ (содержит 0 при α={alpha})", axis=1
    )
    diag['t_intervals'] = ci_df

    # 2. Мультиколлинеарность
    vif_values = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    vif_data = pd.DataFrame({"Признак": X_const.columns, "VIF": vif_values})
    diag['vif'] = vif_data[vif_data['Признак'] != 'const'].sort_values('VIF', ascending=False)
    diag['condition_number'] = np.linalg.cond(X_const.values)

    # 3. Анализ остатков (предпосылки МНК)
    diag['resid_mean'] = np.mean(resid)
    diag['resid_mean_status'] = "Выполнено" if np.abs(diag['resid_mean']) < 1e-7 else "Не выполнено"

    lb = acorr_ljungbox(resid, lags=[10, 20, 100], return_df=True)
    diag['ljungbox'] = lb
    diag['lb_min_p'] = lb['lb_pvalue'].min()
    diag['lb_status'] = "Белый шум (OK)" if diag['lb_min_p'] > alpha else "Автокорреляция (!)"

    lm_stat, bp_p, _, _ = het_breuschpagan(resid, X_const)
    diag['breusch_pagan'] = {
        'LM': lm_stat, 'p_value': bp_p, 
        'status': "Гетероскедастичность (!)" if bp_p < alpha else "Гомоскедастичность (OK)"
    }

    gq_f, gq_p, gq_dir = het_goldfeldquandt(resid, X_const)
    diag['goldfeld_quandt'] = {
        'F': gq_f, 'p_value': gq_p, 'dir': gq_dir,
        'status': "Гетероскедастичность (!)" if gq_p < alpha else "Гомоскедастичность (OK)"
    }

    dw = durbin_watson(resid)
    diag['durbin_watson'] = {
        'DW': dw, 
        'status': "Норма" if 1.5 <= dw <= 2.5 else ("Положительная автокорреляция" if dw < 1.5 else "Отрицательная автокорреляция")
    }

    sh_stat, sh_p = stats.shapiro(resid)
    diag['shapiro'] = {
        'stat': sh_stat, 'p_value': sh_p, 
        'status': "Ненормально (!)" if sh_p < alpha else "Нормально (OK)"
    }

    rs_stat = (np.max(resid) - np.min(resid)) / np.std(resid)
    diag['rs_test'] = {
        'stat': rs_stat, 
        'status': "Выбросы (!)" if rs_stat > 8.0 else ("Узкий разброс" if rs_stat < 3.0 else "Норма")
    }

    jb_stat, jb_p = stats.jarque_bera(resid)
    diag['jarque_bera'] = {
        'stat': jb_stat, 'p_value': jb_p, 
        'skew': stats.skew(resid), 
        'kurt': stats.kurtosis(resid, fisher=False), 
        'status': "Ненормально (!)" if jb_p < alpha else "Нормально (OK)"
    }

    reset_res = reset_ramsey(model, degree=3)
    diag['reset'] = {
        'F': float(reset_res.fvalue), 'p_value': reset_res.pvalue,
        'status': "Ошибка спецификации (!)" if reset_res.pvalue < alpha else "Адекватна (OK)"
    }

    # 4. Тесты Чоу (структурная стабильность)
    diag['chow'] = {}
    masks = {}
    if 'is_apartments' in x_train.columns:
        masks['Апартаменты vs Квартиры'] = x_train['is_apartments'] == 1
    if 'region_of_moscow_CAR' in x_train.columns:
        masks['Центр (ЦАО) vs Окраины'] = x_train['region_of_moscow_CAR'] == 1
    masks['Дорогие vs Дешевые'] = y_train > y_train.median()

    res_total = run_linear_regression({'x_train': x_train, 'x_test': x_train, 'y_train': y_train, 'y_test': y_train}, silent=True)
    sse_total = res_total['sse_sum']
    n_total = len(y_train)
    k_params = x_train.shape[1] + 1

    for name, mask in masks.items():
        try:
            x1, y1 = x_train[mask], y_train[mask]
            x2, y2 = x_train[~mask], y_train[~mask]
            if len(y1) > k_params and len(y2) > k_params:
                x1_f = x1.loc[:, x1.std() > 0]
                x2_f = x2.loc[:, x2.std() > 0]
                sse1 = run_linear_regression({'x_train': x1_f, 'x_test': x1_f, 'y_train': y1, 'y_test': y1}, silent=True)['sse_sum']
                sse2 = run_linear_regression({'x_train': x2_f, 'x_test': x2_f, 'y_train': y2, 'y_test': y2}, silent=True)['sse_sum']
                f_chow = ((sse_total - (sse1 + sse2)) / k_params) / ((sse1 + sse2) / (n_total - 2 * k_params))
                p_chow = 1 - stats.f.cdf(f_chow, k_params, n_total - 2 * k_params)
                diag['chow'][name] = {
                    'F': f_chow, 'p_value': p_chow,
                    'status': "Нарушена (!)" if p_chow < alpha else "Стабильна (OK)",
                    'len1': len(y1), 'len2': len(y2)
                }
        except Exception as e:
            diag['chow'][name] = {'status': f"Ошибка: {str(e)[:30]}..."}

    # 5. Влияние наблюдений и точечный прогноз
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag
    n = len(cooks_d)
    diag['influence'] = {
        'cooks_thr': 4 / n, 'cooks_outliers': int(np.sum(cooks_d > 4 / n)),
        'lev_thr': 3 * (len(model.params) / n), 
        'lev_outliers': int(np.sum(leverage > 3 * len(model.params) / n)),
        'critical_intersection': len(np.intersect1d(np.where(cooks_d > 4 / n)[0], np.where(leverage > 3 * len(model.params) / n)[0]))
    }

    dfbs = influence.dfbetas
    dfb_thr = 2 / np.sqrt(n)
    dfb_report = {}
    for i, name in enumerate(model.params.index):
        impact = np.abs(dfbs[:, i])
        idx = np.where(impact > dfb_thr)[0]
        if len(idx) > 0:
            dfb_report[name] = {
                'count': len(idx), 'pct': (len(idx) / n) * 100,
                'max_val': impact[idx].max(), 'max_idx': int(np.argmax(impact[idx]))
            }
    diag['dfbetas'] = dfb_report

    idx = min(CONFIG["APARTMENT_INDEX"], len(x_test) - 1)
    x0 = sm.add_constant(x_test.iloc[[idx]], has_constant='add')
    x0 = x0[model.model.exog_names]
    pred_obj = model.get_prediction(x0)
    diag['prediction'] = {
        'pred': pred_obj.predicted_mean[0],
        'ci_mean': pred_obj.conf_int(obs=False, alpha=alpha)[0].tolist(),
        'ci_ind': pred_obj.conf_int(obs=True, alpha=alpha)[0].tolist()
    }

    return diag

def collect_correlations(
    data: Dict[str, Any], 
    target_col: str = CONFIG['TARGET_COL']
    ) -> Dict[str, Any]:
    """Рассчитывает парные и частные корреляции, выявляет коллинеарные пары.

    Анализирует линейные связи между признаками и целевой переменной. 
    Частная корреляция вычисляется через обратную матрицу корреляций 
    для оценки изолированного влияния признака при контроле остальных факторов.

    Args:
        data: Контекст пайплайна, содержащий ключ 'final_data' (полный подготовленный DataFrame).
        target_col: Имя целевой переменной для анализа связей.

    Returns:
        Словарь с матрицей корреляций, отчётом по целевой переменной, 
        списком коллинеарных пар и DataFrame с частными корреляциями.
    """
    final_data = data['final_data']
    numeric_df = final_data.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    price_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
    pairs_report = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i):
            if cols[i] == target_col or cols[j] == target_col: 
                continue
            val = corr_matrix.iloc[i, j]
            if abs(val) > CONFIG["CORR_THRESHOLD"]:
                pairs_report.append(f"[!] {cols[i]} и {cols[j]}: {val:.3f}")
                
    try:
        precision = pd.DataFrame(
            np.linalg.inv(corr_matrix.values), 
            index=corr_matrix.columns, 
            columns=corr_matrix.columns
        )
        ext_report = []
        for col in corr_matrix.columns:
            if col == target_col: 
                continue
            r_part = -precision.loc[col, target_col] / np.sqrt(precision.loc[col, col] * precision.loc[target_col, target_col])
            ext_report.append({
                'Признак': col, 
                'Парная': corr_matrix.loc[col, target_col], 
                'Частная': r_part, 
                'Дельта': abs(corr_matrix.loc[col, target_col] - r_part)
            })
        ext_df = pd.DataFrame(ext_report).sort_values('Частная', ascending=False)
    except np.linalg.LinAlgError:
        ext_df = pd.DataFrame()
        
    return {'matrix': corr_matrix, 'price_corr': price_corr, 'pairs': pairs_report, 'extended': ext_df}

def format_diagnostics_report(
    diag: Dict[str, Any], 
    alpha: float = CONFIG["ALPHA"]
    ) -> None:
    """Формирует и выводит в консоль академический отчёт по экономической диагностике.

    Структурирует результаты тестов по логическим блокам: значимость, 
    мультиколлинеарность, предпосылки МНК, спецификация, влияние наблюдений 
    и прогностическая способность. Включает развёрнутые интерпретации.

    Args:
        diag: Словарь с результатами, собранными функцией `collect_diagnostic_data`.
        alpha: Уровень значимости, использованный при расчётах.
    """
    print("\n[1] ПРОВЕРКА ЗНАЧИМОСТИ\n" + "-"*30)
    f = diag['fisher']
    print(f"F-TEST (Значимость уравнения в целом)\nF-расчетное: {f['F_calc']:.2f} | F-критическое: {f['F_crit']:.2f}\np-value: {f['p_value']:.4e} (Порог: {alpha})\nВердикт: Модель {f['status']}")

    t_crit = diag['t_crit']
    print(f"\nt-TEST (Значимость коэффициентов) | t_crit: {t_crit:.3f}")
    print(diag['t_test'].to_string())

    ci_df = diag['t_intervals'].copy()
    ci_df.columns = ['Коэффициент', f'Нижняя граница ({alpha/2*100:.1f}%)', f'Верхняя граница ({(1-alpha/2)*100:.1f}%)', 'Надежен?']
    print(f"\n[!] Анализ {int((1-alpha)*100)}% доверительных интервалов (α={alpha}):")
    print("-" * 80)
    print(ci_df.to_string())
    print("-" * 80)

    print("\n[2] МУЛЬТИКОЛЛИНЕАРНОСТЬ\n" + "-"*30)
    print(diag['vif'].assign(VIF=lambda x: x.VIF.round(2)).to_string(index=False))
    print(f"\n[OK] Все значения VIF в норме (ниже {CONFIG['VIF_THRESHOLD']}).\n")
    cn = diag['condition_number']
    cn_txt = "КРИТИЧЕСКАЯ мультиколлинеарность." if cn > 1000 else "Умеренная мультиколлинеарность." if cn > 30 else "Матрица стабильна."
    print(f"CONDITION NUMBER: {cn:.2f}\nРезультат: {cn_txt}")

    print("\n[3] АНАЛИЗ ОСТАТКОВ (ПРЕДПОСЫЛКИ МНК)\n" + "-"*30)
    resid_status = "Условие Гаусса-Маркова выполнено (среднее равно нулю)." if "Выполнено" in diag['resid_mean_status'] else "Условие НЕ выполнено."
    print(f"Нулевое среднее: {diag['resid_mean']:.4e}\nРезультат: {resid_status}")
    
    lb = diag['ljungbox']
    print(f"\nСлучайность (Ljung-Box) | p-min: {diag['lb_min_p']:.4f}\nВердикт: {diag['lb_status']}")
    print(lb.to_string(index=False))
    if diag['lb_min_p'] > alpha:
        print("     Модель полностью извлекла информацию из признаков.")

    bp = diag['breusch_pagan']
    print(f"\n---------- ТЕСТ БРЕУША-ПАГАНА ----------\nLagrange Multiplier stat: {bp['LM']:.4f}\np-value: {bp['p_value']:.4e}")
    if bp['p_value'] < alpha:
        print("РЕЗУЛЬТАТ: Обнаружена гетероскедастичность (H0 отвергнута).\nВЛИЯНИЕ: Оценки коэффициентов несмещены, но их стандартные ошибки ненадёжны.\nСОВЕТ: Используйте робастные стандартные ошибки (HC3) для проверки значимости.")
    else:
        print("РЕЗУЛЬТАТ: Гетероскедастичность не обнаружена (H0 подтверждена).")

    gq = diag['goldfeld_quandt']
    dirs = {'increasing': 'Рост дисперсии', 'decreasing': 'Убывание дисперсии', 'two-sided': 'Двустороннее изменение'}
    print(f"\n---------- ТЕСТ ГОЛДФЕЛДА-КВАНДТА ----------\nF-статистика: {gq['F']:.4f}\np-value: {gq['p_value']:.4e}\nНаправление: {dirs.get(gq['dir'], gq['dir'])}")
    if gq['p_value'] < alpha:
        print("Результат: Гетероскедастичность ОБНАРУЖЕНА.\nРекомендация: Проверьте структуру данных или используйте робастные ошибки (HC).")
    else:
        print("Результат: Гомоскедастичность подтверждена (H0 не отвергается).\nВывод: Дисперсия остатков стабильна на всей выборке.")

    dw = diag['durbin_watson']
    print(f"\n---------- ТЕСТ ДАРБИНА-УОТСОНА ----------\nЗначение DW-статистики: {dw['DW']:.3f}")
    if 1.5 <= dw['DW'] <= 2.5:
        print("Результат: Автокорреляция не обнаружена (в пределах нормы).\nВывод: Остатки распределены случайно, условие независимости соблюдено.")
    elif dw['DW'] < 1.5:
        print("Результат: Обнаружена ПОЛОЖИТЕЛЬНАЯ автокорреляция.\nВывод: Ошибки имеют тенденцию сохранять знак. Возможен пропущенный тренд.")
    else:
        print("Результат: Обнаружена ОТРИЦАТЕЛЬНАЯ автокорреляция.\nВывод: Ошибки слишком часто меняют знак. Возможна избыточная подгонка.")

    print(f"\nНормальность (Shapiro): p={diag['shapiro']['p_value']:.4e} | {diag['shapiro']['status']}")
    if diag['shapiro']['p_value'] < alpha:
        print("Примечание: На большой выборке тест может быть избыточно строгим. Рекомендуется проверить QQ-plot.")

    rs = diag['rs_test']
    print(f"\n---------- RS-КРИТЕРИЙ (Range/Std) ----------\nСтатистика (q): {rs['stat']:.2f}")
    if rs['stat'] > 8.0: 
        print("РЕЗУЛЬТАТ: Обнаружены СИЛЬНЫЕ ВЫБРОСЫ (тяжелые хвосты).\nСовет: Проверьте данные на наличие аномалий или используйте Robust регрессию.")
    elif rs['stat'] < 3.0: 
        print("РЕЗУЛЬТАТ: Слишком узкий разброс (короткие хвосты).")
    else: 
        print("РЕЗУЛЬТАТ: Соотношение размаха к отклонению в пределах нормы.")

    jb = diag['jarque_bera']
    print(f"\n---------- ТЕСТ ЖАКА-БЕРА (JARQUE-BERA) ----------\nСтатистика JB: {jb['stat']:.2f}\np-value: {jb['p_value']:.4e}\nАсимметрия (Skew): {jb['skew']:.3f}\nЭксцесс (Kurt): {jb['kurt']:.3f}")
    if jb['p_value'] < alpha:
        reasons = [r for r, cond in [("сильная асимметрия", abs(jb['skew'])>0.5), ("нетипичная острота пика/хвостов", abs(jb['kurt']-3)>1)] if cond]
        print(f"Результат: Распределение НЕ нормальное.\nПричина: {', '.join(reasons) if reasons else 'отклонение формы'}.")
    else:
        print("Результат: Распределение близко к нормальному.")

    print("\n[4] ТЕСТЫ СПЕЦИФИКАЦИИ\n" + "-"*30)
    rst = diag['reset']
    print(f"RESET TEST: F={rst['F']:.4f}, p={rst['p_value']:.4e} | {rst['status']}")
    if rst['p_value'] < alpha:
        print("Вывод: Линейная форма модели неполна. Возможны пропущенные нелинейные члены, взаимодействия или значимые переменные.")

    print("\nТесты Чоу (Структурная стабильность):")
    for name, res in diag['chow'].items():
        print(f"\n!!!!!!!!!!!!!!!!!!!! ТЕСТ ЧОУ: {name} !!!!!!!!!!!!!!!!!!!!")
        print(f"Размер группы 1: {res.get('len1','?')} | Размер группы 2: {res.get('len2','?')}")
        print(f"F-расчетное: {res['F']:.4f} | p-value: {res['p_value']:.4e}")
        print(f">>> ВЫВОД: {'Структурная стабильность НАРУШЕНА.' if res['p_value'] < alpha else 'Структура СТАБИЛЬНА.'}")
        if res['p_value'] < alpha: 
            print("    Рекомендация: Постройте отдельные модели для подгрупп.")

    print("\n[5] ВЛИЯТЕЛЬНЫЕ НАБЛЮДЕНИЯ\n" + "-"*30)
    inf = diag['influence']
    print(f"1. Расстояние Кука (Cook's D):\n   - Порог (4/n): {inf['cooks_thr']:.5f}\n   - Найдено влиятельных объектов: {inf['cooks_outliers']}")
    print(f"\n2. Рычаги (Leverage):\n   - Порог (3*k/n): {inf['lev_thr']:.5f}\n   - Аномальных X-характеристик: {inf['lev_outliers']}")
    if inf['critical_intersection'] > 0:
        print(f"\n[!!!] КРИТИЧЕСКИЕ ТОЧКИ: {inf['critical_intersection']} объектов одновременно имеют аномальный X и сильно искажают модель.")

    print(f"\n=============== АНАЛИЗ DFBETAS (Влияние на конкретные веса) ===============")
    print(f"Порог (2/sqrt(n)): {2/np.sqrt(len(diag['vif'])+1):.4f}")
    for feat, info in diag.get('dfbetas', {}).items():
        print(f"\n[!] Признак '{feat}':\n    - Проблемных точек: {info['count']} ({info['pct']:.1f}%)\n    - Макс. влияние: {info['max_val']:.4f} (индекс {info['max_idx']})")
    if not diag.get('dfbetas'): 
        print("Результат: Влиятельных наблюдений для весов признаков не обнаружено.")

    print("\n[6] ПРОГНОСТИЧЕСКАЯ СПОСОБНОСТЬ\n" + "-"*30)
    prd = diag['prediction']
    print(f"Точечный прогноз цены: {prd['pred']:,.0f} руб.")
    print(f"\n{int((1-alpha)*100)}% ДИ для СРЕДНЕГО значения (рыночная норма):\n  [{prd['ci_mean'][0]:,.0f} - {prd['ci_mean'][1]:,.0f}] руб.")
    print(f"\n{int((1-alpha)*100)}% ДИ для ИНДИВИДУАЛЬНОГО значения (вилка для объекта):\n  [{max(0,prd['ci_ind'][0]):,.0f} - {prd['ci_ind'][1]:,.0f}] руб.")
    
    print("\n" + "="*50 + "\nДИАГНОСТИКА ЗАВЕРШЕНА\n" + "="*50)

def collect_robustness_comparison(
    data: Dict[str, Any], 
    alpha: float = CONFIG["ALPHA"]
    ) -> pd.DataFrame:
    """Сравнивает статистическую значимость признаков для обычной и робастной (HC3) моделей.

    Выявляет признаки, теряющие или приобретающие значимость при переходе 
    к устойчивым стандартным ошибкам, что критично для верификации выводов.

    Args:
        data: Контекст с ключами 'res_raw' (обычная модель) и 'res_robust' (робастная).
        alpha: Порог значимости для классификации изменений.

    Returns:
        DataFrame с колонками: Признак, P-value (Обычный), P-value (HC3), Изменение, Вердикт.
    """
    res_raw = data['res_raw']
    res_robust = data['res_robust']
    
    interp_raw = res_raw.get('Interpretation')
    interp_rob = res_robust.get('Interpretation')
    if interp_raw is None or interp_rob is None: 
        return pd.DataFrame()

    comp = pd.DataFrame({
        'Признак': interp_raw['Признак'],
        'P-value (Обычный)': interp_raw['P-value'],
        'P-value (HC3)': interp_rob['P-value']
    })
    comp['Изменение'] = comp['P-value (HC3)'] - comp['P-value (Обычный)']

    lost = comp[(comp['P-value (Обычный)'] < alpha) & (comp['P-value (HC3)'] > alpha)]
    gained = comp[(comp['P-value (Обычный)'] > alpha) & (comp['P-value (HC3)'] < alpha)]

    comp['Вердикт'] = np.where(
        comp.index.isin(lost.index), 'Потеря значимости',
        np.where(comp.index.isin(gained.index), 'Приобретение', 'Стабильно')
    )
    return comp

def format_correlations_report(
    corr_data: Dict[str, Any], 
    alpha: float = CONFIG["ALPHA"]
    ) -> None:
    """Формирует текстовый отчёт по корреляционному анализу.

    Выводит матрицу парных корреляций, ранжирует признаки по связи с таргетом, 
    выявляет коллинеарные пары и сравнивает парные/частные корреляции.

    Args:
        corr_data: Словарь с результатами `collect_correlations`.
        alpha: Уровень значимости (используется для порогов в выводе).
    """
    print("\n" + "="*30 + " КОРРЕЛЯЦИОННЫЙ АНАЛИЗ " + "="*30+"\n")
    print("\nМАТРИЦА ПАРНЫХ КОРРЕЛЯЦИЙ:\n" + corr_data['matrix'].round(2).to_string())
    print(f"\nСвязь признаков с целевой переменной '{CONFIG['TARGET_COL']}':")
    for feat, val in corr_data['price_corr'].items():
        strength = "Сильная" if abs(val) > 0.7 else "Средняя" if abs(val) > 0.3 else "Слабая"
        print(f"  {val:>6.3f} | {feat:<25} ({strength})")
        
    print("\n" + "-"*15 + f" ПОИСК КОЛЛИНЕАРНЫХ ПАР (>{CONFIG['CORR_THRESHOLD']}) " + "-"*15)
    print("\n".join(corr_data['pairs']) if corr_data['pairs'] else "Сильных межфакторных связей не обнаружено.")
    
    if not corr_data['extended'].empty:
        print("\n" + "="*15 + " СРАВНЕНИЕ ПАРНОЙ И ЧАСТНОЙ КОРРЕЛЯЦИИ " + "="*15)
        print(corr_data['extended'].to_string(index=False, formatters={'Парная': '{:,.3f}'.format, 'Частная': '{:,.3f}'.format, 'Дельта': '{:,.3f}'.format}))
        distorted = corr_data['extended'][corr_data['extended']['Дельта'] > 0.4].sort_values('Дельта', ascending=False)
        if not distorted.empty:
            print(f"\n[!] ОБНАРУЖЕНО СИЛЬНОЕ ИСКАЖЕНИЕ (Дельта > 0.4):")
            for _, row in distorted.iterrows():
                print(f"    • '{row['Признак']}': упала с {row['Парная']:.2f} до {row['Частная']:.2f}")

def format_robustness_report(
    comp_df: pd.DataFrame, 
    alpha: float = CONFIG["ALPHA"]
    ) -> None:
    """Формирует отчёт о сравнении устойчивости коэффициентов к гетероскедастичности.

    Анализирует, как переход на робастные ошибки (HC3) меняет статус значимости 
    предикторов, и классифицирует изменения по категориям.

    Args:
        comp_df: DataFrame сравнения, сформированный `collect_robustness_comparison`.
        alpha: Порог значимости для интерпретации вердиктов.
    """
    print("\n" + "="*15 + " СРАВНЕНИЕ ЗНАЧИМОСТИ ПРИЗНАКОВ " + "="*15)
    print(comp_df.to_string(index=False))
    if not comp_df.empty:
        lost = comp_df[comp_df['Вердикт'] == 'Потеря значимости']['Признак'].tolist()
        gained = comp_df[comp_df['Вердикт'] == 'Приобретение']['Признак'].tolist()
        if lost: 
            print(f"\n[!] ПОТЕРЯ ЗНАЧИМОСТИ (HC3): {lost}")
        if gained: 
            print(f"\n[+] ПРИОБРЕТЕНИЕ ЗНАЧИМОСТИ (HC3): {gained}")
        if not lost and not gained: 
            print("\n[OK] Изменений в составе значимых признаков не обнаружено.")


# ==============================================================================
# 7. ОТБОР ПРИЗНАКОВ И ИТЕРАТИВНОЕ УЛУЧШЕНИЕ
# ==============================================================================

def get_feature_selection_advice(
    data: Dict[str, Any], 
    alpha: float = CONFIG["ALPHA"], 
    priority_threshold: int = CONFIG["priority_threshold"], 
    silent: bool = False
    ) -> Tuple[pd.DataFrame, List[str]]:
    """Проводит многокритериальный аудит признаков для отбора кандидатов на удаление.

    Оценивает каждый регрессор по четырём эконометрическим критериям:
    1. Статистическая значимость (p-value > alpha).
    2. Мультиколлинеарность (VIF > порог).
    3. Надёжность доверительного интервала (пересечение с нулём).
    4. Частная корреляция с целевой переменной (близка к нулю).
    На основе накопленного балла признаки классифицируются по приоритету исключения.

    Args:
        data: Контекст пайплайна, содержащий ключи:
              'model_result' (результат обучения), 'x_train', 'final_data'.
        alpha: Уровень значимости для проверки гипотез.
        priority_threshold: Уровень строгости отбора (1=только высокий, 2=высокий+средний, 3=все).
        silent: Флаг подавления вывода в консоль.

    Returns:
        Кортеж из (DataFrame с отчётом аудита, Список признаков-кандидатов на удаление).
    """
    model_result = data['model_result']
    x_train = data['x_train']
    final_data = data['final_data']

    interp = model_result.get('Interpretation')
    if interp is None:
        if not silent:
            print("[!] Ошибка: В результатах модели не найдена таблица Interpretation.")
        return pd.DataFrame(), []

    model = model_result['model_object']
    current_metrics = {'Adj.R2': model.rsquared_adj, 'AIC': model.aic, 'BIC': model.bic}
    conf_int_df = model.conf_int()

    X_const = sm.add_constant(x_train)
    vif_series = pd.Series(
        [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])],
        index=X_const.columns
    ).drop('const', errors='ignore')

    numeric_df = final_data.select_dtypes(include=[np.number])
    corr_mat = numeric_df.corr()
    try:
        precision_df = pd.DataFrame(
            np.linalg.inv(corr_mat.values + np.eye(corr_mat.shape[0]) * 1e-9),
            index=corr_mat.index, columns=corr_mat.columns
        )
    except np.linalg.LinAlgError:
        precision_df = pd.DataFrame()

    report = []
    for _, row in interp.iterrows():
        feat = row['Признак']
        if feat == 'const':
            continue

        p = row['P-value']
        ci_l, ci_u = conf_int_df.loc[feat]
        reasons, score = [], 0

        if p > alpha:
            reasons.append(f"p={p:.3f} > α")
            score += 1.0

        v = vif_series.get(feat, 0)
        if v > CONFIG["VIF_THRESHOLD"]:
            reasons.append(f"VIF={v:.1f}")
            score += 0.7

        if ci_l <= 0 <= ci_u:
            reasons.append("ДИ содержит 0")
            score += 0.3

        r_part = np.nan
        if not precision_df.empty and feat in precision_df.index and CONFIG["TARGET_COL"] in precision_df.columns:
            p_ij = precision_df.loc[feat, CONFIG["TARGET_COL"]]
            p_ii = precision_df.loc[feat, feat]
            p_jj = precision_df.loc[CONFIG["TARGET_COL"], CONFIG["TARGET_COL"]]
            r_part = -p_ij / np.sqrt(p_ii * p_jj)
            if abs(r_part) < 0.05:
                reasons.append(f"r_part={r_part:.3f}≈0")
                score += 0.5

        if reasons:
            report.append({
                'Признак': feat,
                'Причины': "; ".join(reasons),
                'Приоритет': 'ВЫСОКИЙ' if score >= 1.5 else 'СРЕДНИЙ' if score >= 0.8 else 'НИЗКИЙ',
                'Прогноз Adj.R2': "Улучшит" if p > alpha else "Рискованно",
                'Прогноз AIC': "Снизит" if p > alpha else "Повысит",
                'VIF': round(v, 2),
                'r_part': round(r_part, 3) if not np.isnan(r_part) else "N/A"
            })

    df = pd.DataFrame(report)
    priority_map = {1: ['ВЫСОКИЙ'], 2: ['ВЫСОКИЙ', 'СРЕДНИЙ'], 3: ['ВЫСОКИЙ', 'СРЕДНИЙ', 'НИЗКИЙ']}
    target_priorities = priority_map.get(priority_threshold, ['ВЫСОКИЙ'])

    if not silent:
        print("\n" + "="*80 + "\nЭТАП: АУДИТ И СЕЛЕКЦИЯ ПРИЗНАКОВ\n" + "="*80)
        print(f"ТЕКУЩЕЕ СОСТОЯНИЕ: Adj.R2: {current_metrics['Adj.R2']:.4f} | AIC: {current_metrics['AIC']:.1f}")
        print(f"Режим отбора: Уровень {priority_threshold} (Включает: {', '.join(target_priorities)})")
        if not df.empty:
            df['sort'] = df['Приоритет'].map({'ВЫСОКИЙ': 0, 'СРЕДНИЙ': 1, 'НИЗКИЙ': 2})
            print(df.sort_values('sort').drop(columns='sort').to_string(index=False))
            candidates = df[df['Приоритет'].isin(target_priorities)]['Признак'].tolist()
            if candidates:
                print(f"\n[РЕКОМЕНДАЦИЯ: Исключить {len(candidates)} признаков: {candidates}")
        else:
            print("\n[OK] Все признаки качественны.")

    return df, df[df['Приоритет'].isin(target_priorities)]['Признак'].tolist() if not df.empty else []

def run_iterative_refinement(
    data: Dict[str, Any], 
    candidates_to_drop: List[str], 
    cov_type: str = 'HC3', 
    silent: bool = False
) -> Dict[str, Any]:
    """Выполняет пошаговое исключение признаков с отслеживанием метрик качества.

    Последовательно удаляет кандидатов из обучающей выборки, заново обучает модель
    и фиксирует изменение Adj.R2, AIC и F-статистики. Процесс останавливается после
    удаления всех кандидатов или при критическом ухудшении информационного критерия.

    Args:
        data: Контекст пайплайна с ключами 'x_train', 'x_test', 'y_train', 'y_test'.
        candidates_to_drop: Список признаков, подлежащих последовательному удалению.
        cov_type: Тип ковариационной матрицы для расчёта стандартных ошибок.
        silent: Флаг подавления промежуточного вывода.

    Returns:
        Словарь с результатами: 'refined_x_train', 'refined_x_test', 'history' (DataFrame),
        'last_model' (результат последней итерации).
    """
    x_train, x_test = data['x_train'].copy(), data['x_test'].copy()
    y_train, y_test = data['y_train'], data['y_test']
    history = []
    drop_set = set(candidates_to_drop)

    if not silent:
        print("\n" + "="*80 + "\nЭТАП: ИТЕРАТИВНОЕ УЛУЧШЕНИЕ МОДЕЛИ\n" + "="*80)
        print(f"Исходно признаков: {x_train.shape[1]} | Кандидаты: {len(drop_set)}")

    base_model = run_linear_regression(
        {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}, 
        cov_type=cov_type, 
        silent=True
    )
    m = base_model['metrics']
    history.append({
        'Удален': 'НИКОГО',
        'Adj.R2': m.get('Adj_R2', base_model['model_object'].rsquared_adj),
        'AIC': base_model['model_object'].aic,
        'F-pvalue': base_model['model_object'].f_pvalue
    })

    last_model = base_model
    valid_candidates = [f for f in drop_set if f in x_train.columns]

    for feat in valid_candidates:
        x_train = x_train.drop(columns=[feat])
        x_test = x_test.drop(columns=[feat])

        step_model = run_linear_regression(
            {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}, 
            cov_type=cov_type, 
            silent=True
        )
        last_model = step_model
        m_step = step_model['metrics']
        history.append({
            'Удален': feat,
            'Adj.R2': m_step.get('Adj_R2', step_model['model_object'].rsquared_adj),
            'AIC': step_model['model_object'].aic,
            'F-pvalue': step_model['model_object'].f_pvalue
        })

        if not silent and len(history) > 1:
            prev_aic = history[-2]['AIC']
            curr_aic = history[-1]['AIC']
            if curr_aic < prev_aic:
                status = "[+] AIC улучшился"
            elif curr_aic > prev_aic + 2:
                status = "[!] AIC вырос (модель ухудшилась)"
            else:
                status = "[~] Нейтрально"
            print(f"[-] {feat:<20} | Adj.R2: {history[-1]['Adj.R2']:.4f} | AIC: {history[-1]['AIC']:.1f} {status}")

    history_df = pd.DataFrame(history)
    if not silent:
        print("\n" + "-"*30 + " ИТОГИ ОПТИМИЗАЦИИ " + "-"*30)
        print(history_df.to_string(index=False))
        delta = history[0]['AIC'] - history[-1]['AIC']
        print(f"\n{'AIC улучшен на' if delta > 0 else 'AIC не улучшен'} {abs(delta):.1f}\n" + "-"*80)

    return {
        'refined_x_train': x_train,
        'refined_x_test': x_test,
        'history': history_df,
        'last_model': last_model
    }

def format_selection_report(
        selection_df: pd.DataFrame, 
        candidates: List[str], 
        priority_threshold: int = CONFIG["priority_threshold"], 
        alpha: float = CONFIG["ALPHA"]
        ) -> None:
    """Формирует текстовый отчёт по результатам аудита и селекции признаков.

    Выводит отсортированный список кандидатов на удаление с указанием причин,
    приоритета и прогнозируемого влияния на информационные критерии.

    Args:
        selection_df: DataFrame с результатами `get_feature_selection_advice`.
        candidates: Отфильтрованный список признаков для исключения.
        priority_threshold: Использованный уровень строгости отбора.
        alpha: Уровень значимости.
    """
    print("\n" + "="*80 + "\nЭТАП: АУДИТ И СЕЛЕКЦИЯ ПРИЗНАКОВ\n" + "="*80)
    print(f"Режим отбора: Уровень {priority_threshold}")

    if not selection_df.empty:
        df_print = selection_df.copy()
        if 'Приоритет' in df_print.columns:
            df_print['sort'] = df_print['Приоритет'].map({'ВЫСОКИЙ': 0, 'СРЕДНИЙ': 1, 'НИЗКИЙ': 2})
            print(df_print.sort_values('sort').drop(columns='sort').to_string(index=False))
        else:
            print(df_print.to_string(index=False))
        if candidates:
            print(f"\n[РЕКОМЕНДАЦИЯ: Исключить {len(candidates)} признаков: {candidates}")
    else:
        print("\n[OK] Все признаки качественны. Кандидатов на удаление нет.")

def format_refinement_log(
    history_df: pd.DataFrame
    ) -> None:
    """Выводит пошаговую историю изменения метрик при итеративном отборе.

    Форматирует DataFrame с историей шагов, отображая изменение AIC и Adj.R2
    на каждом этапе удаления признака.

    Args:
        history_df: DataFrame с колонками 'Удален', 'Adj.R2', 'AIC', 'F-pvalue'.
    """
    print("\n[1] ЛОГ ИТЕРАТИВНОГО УЛУЧШЕНИЯ (ШАГОВОЙ ОТБОР)")
    print("-" * 60)
    print(history_df.to_string(index=False))
    delta = history_df.iloc[0]['AIC'] - history_df.iloc[-1]['AIC']
    status = "AIC улучшен на" if delta > 0 else "AIC не улучшен"
    print(f"\n{status} {abs(delta):.1f}")
    print("="*80)


# ==============================================================================
# 8. ИТОГОВОЕ СРАВНЕНИЕ И ЭКСПОРТ ОТЧЁТОВ
# ==============================================================================

def display_comprehensive_models_comparison(
    data: Dict[str, Any]
    ) -> pd.DataFrame:
    """Формирует и выводит сводную таблицу сравнения всех обученных моделей.

    Динамически агрегирует архитектурные параметры, статистические метрики
    statsmodels, результаты машинного обучения (MAE, RMSE, AIC и др.),
    а также показатели мультиколлинеарности. Автоматически определяет
    лидера по каждому из ключевых критериев качества.

    Args:
        data: Контекст пайплайна, содержащий ключи:
              'models_results' (словарь результатов обучения),
              'x_train_dict' (обучающие выборки для расчёта VIF),
              'alpha' (уровень значимости, по умолчанию из CONFIG).

    Returns:
        DataFrame с упорядоченными метриками и диагностикой по каждой модели.
    """
    models_results_dict = data['models_results']
    x_train_dict = data.get('x_train_dict')
    alpha = data.get('alpha', CONFIG["ALPHA"])

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 300)

    rows = []
    print("\n" + "="*180)
    print("ПОЛНОЕ СРАВНЕНИЕ МОДЕЛЕЙ:")
    print("="*180)

    for name, res in models_results_dict.items():
        if not isinstance(res, dict):
            continue

        model = res.get('model_object')
        m = res.get('metrics', {})
        interp = res.get('Interpretation')
        row = {'Модель': name.upper()}

        # 1. Архитектура и гиперпараметры
        row['Алгоритм'] = model.__class__.__name__ if model else '—'
        row['Cov_Type'] = res.get('cov_type', '—')
        row['Признаков'] = res.get('n_features', 0)
        if hasattr(model, 'n_estimators'):
            row['Estimators'] = model.n_estimators
        if hasattr(model, 'get_depth'):
            row['MaxDepth'] = model.get_depth()
        if hasattr(model, 'get_params'):
            p = model.get_params()
            if 'alpha' in p:
                row['Alpha'] = p['alpha']

        # 2. Статистика statsmodels
        if hasattr(model, 'rsquared'):
            row['R2'] = round(model.rsquared, 4)
            row['Adj_R2'] = round(model.rsquared_adj, 4)
            row['F-стат.'] = f"{model.fvalue:.1f}"
            row['F-pval'] = f"{model.f_pvalue:.2e}"
            row['LogLik'] = round(model.llf, 1)
            row['AIC'] = round(model.aic, 1)
            row['BIC'] = round(model.bic, 1)
            row['Параметров'] = len(model.params)
            try:
                row['p(const)'] = f"{model.pvalues['const']:.3f}"
            except KeyError:
                row['p(const)'] = '—'
        else:
            for k in ['R2', 'Adj_R2', 'F-стат.', 'F-pval', 'LogLik', 'AIC', 'BIC', 'Параметров', 'p(const)']:
                row[k] = '—'

        # 3. Динамический сбор всех метрик из словаря
        for k, v in m.items():
            if v is None or (isinstance(v, float) and pd.isna(v)):
                row[k] = '—'
            elif isinstance(v, float):
                row[k] = round(v, 4) if abs(v) < 1e4 else f"{v:.2e}"
            elif isinstance(v, int):
                row[k] = f"{v:,}"
            else:
                row[k] = v

        # 4. Значимость признаков
        if interp is not None and hasattr(interp, 'empty') and not interp.empty and 'P-value' in interp.columns:
            feats = interp[interp['Признак'] != 'const']
            total, sig = len(feats), len(feats[feats['P-value'] < alpha])
            row['Значимые'] = f"{sig}/{total}"
        else:
            row['Значимые'] = 'Black Box / N/A'

        # 5. Коллинеарность
        if x_train_dict and name in x_train_dict:
            X = sm.add_constant(x_train_dict[name])
            try:
                vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1]) if X.columns[i] != 'const']
                row['Max VIF'] = round(max(vifs), 2)
                row['Cond.Num'] = f"{np.linalg.cond(X.values):.2e}"
            except Exception:
                row['Max VIF'] = row['Cond.Num'] = '—'
        else:
            row['Max VIF'] = row['Cond.Num'] = '—'

        rows.append(row)

    df = pd.DataFrame(rows).fillna('—')

    # Переименование для стандартизации
    df = df.rename(columns={
        'R2': 'R²', 'Adj_R2': 'Adj. R²', 'MAPE': 'MAPE (%)',
        'Normality_Status': 'Нормальность', 'R2_log': 'R² (log)', 'Adj_R2_log': 'Adj.R² (log)'
    })

    # Формирование строгого порядка колонок
    priority_cols = [
        'Модель', 'Алгоритм', 'Cov_Type', 'Признаков',
        'R²', 'Adj. R²', 'R² (log)', 'Adj.R² (log)', 'F-стат.', 'F-pval', 'LogLik', 'AIC', 'BIC', 'Параметров', 'p(const)',
        'MAE', 'RMSE', 'MAPE (%)', 'MSLE', 'MSE', 'Huber', 'Skewness', 'Kurtosis', 'Нормальность', 'Значимые',
        'Estimators', 'MaxDepth', 'Alpha'
    ]
    diag_cols = ['Значимые', 'Max VIF', 'Cond.Num']

    fixed_ordered = [c for c in priority_cols if c in df.columns]
    dynamic_cols = [c for c in df.columns if c not in fixed_ordered and c not in diag_cols]
    fixed_diag = [c for c in diag_cols if c in df.columns]

    final_cols = fixed_ordered + dynamic_cols + fixed_diag
    df = df[final_cols]

    print(df.to_string(index=False))
    print("-" * 180)

    # Автоматический анализ победителей
    print("АВТОМАТИЧЕСКИЙ ВЫБОР ОПТИМАЛЬНОЙ СПЕЦИФИКАЦИИ:")
    priority_map = {
        'Информативность (AIC ↓)': 'AIC',
        'Прогностическая точность (MAPE ↓)': 'MAPE (%)',
        'Абсолютная ошибка (RMSE ↓)': 'RMSE',
        'Средняя ошибка (MAE ↓)': 'MAE',
        'Объясняющая сила (R² ↑)': 'R²',
        'Объясняющая сила (Adj.R² ↑)': 'Adj. R²',
        'Робастность (Huber ↓)': 'Huber',
        'Логарифмическая точность (MSLE ↓)': 'MSLE'
    }

    for label, col in priority_map.items():
        if col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            if numeric_col.notna().any():
                idx = numeric_col.idxmax() if '↑' in label else numeric_col.idxmin()
                if pd.notna(idx):
                    print(f"{label}: {df.loc[idx, 'Модель']} ({df.loc[idx, col]})")
    return df

def export_linear_analysis_report(
    data: Dict[str, Any]
    ) -> None:
    """Генерирует Отчёт 1: Полный эконометрический анализ базовой линейной модели.

    Формирует структурированный текстовый файл, содержащий метрики качества,
    ANOVA-таблицу, summary statsmodels, математическое уравнение, корреляционный
    анализ, диагностику остатков, сравнение обычной и робастной (HC3) моделей,
    а также аудит и селекцию признаков. Использует механизм захвата stdout.

    Args:
        data: Контекст пайплайна с ключами:
              'models_results', 'diagnostics', 'corr_data', 'robustness_df',
              'selection_report', 'candidates', 'alpha'.
    """
    models_results = data['models_results']
    diagnostics = data['diagnostics']
    corr_data = data['corr_data']
    robustness_df = data['robustness_df']
    selection_report = data['selection_report']
    candidates = data['candidates']
    alpha = data.get('alpha', CONFIG["ALPHA"])

    def _generate() -> None:
        print("="*100 + "\nОТЧЁТ 1: ЛИНЕЙНАЯ МОДЕЛЬ И ЕЁ ПОЛНЫЙ ЭКОНОМЕТРИЧЕСКИЙ АНАЛИЗ\n" + "="*100)
        print_metrics_template(models_results['linear']['metrics'], "Линейная Регрессия (исходная)")
        print("\n\n[1] МЕТРИКИ И SUMMARY ЛИНЕЙНОЙ МОДЕЛИ")
        print(f"\n{'='*20} ANOVA TABLE {'='*20}")
        print(models_results['linear']['anova_table'].to_string(index=False))
        print(models_results['linear']['model_object'].summary())
        print("\n\n" + "-"*30 + " МАТЕМАТИЧЕСКОЕ УРАВНЕНИЕ МОДЕЛИ " + "-"*30)
        print(models_results['linear'].get('formula', 'Формула не найдена'))

        if 'Interpretation' in models_results['linear']:
            print("\n" + "-"*30 + " ИНТЕРПРЕТАЦИЯ КОЭФФИЦИЕНТОВ " + "-"*30)
            print(models_results['linear']['Interpretation'].sort_values(by='Коэффициент', ascending=False).to_string(index=False))

        print("\n\n[2] КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
        format_correlations_report(corr_data, alpha)

        print("\n\n[3] ДИАГНОСТИКА БАЗОВОЙ ЛИНЕЙНОЙ МОДЕЛИ")
        format_diagnostics_report(diagnostics, alpha)

        print("\n\n[4] СРАВНЕНИЕ ОБЫЧНОЙ И РОБАСТНОЙ (HC3) МОДЕЛЕЙ")
        format_robustness_report(robustness_df, alpha)

        print("\n\n[5] АУДИТ И СЕЛЕКЦИЯ ПРИЗНАКОВ")
        format_selection_report(selection_report, candidates, alpha=alpha)

        print("\n" + "="*100 + "\nКОНЕЦ ОТЧЁТА 1")

    _capture_output_and_save("text\\1_Линейная_модель_и_анализ.txt", _generate)

def export_refined_model_report(
    data: Dict[str, Any]
    ) -> None:
    """Генерирует Отчёт 2: Анализ итерационного улучшения и очищенной модели.

    Включает лог пошагового отбора признаков (изменение AIC/Adj.R2),
    детальную статистику финальной (Refined) модели, её математическое уравнение,
    интерпретацию коэффициентов и техническую диагностику остатков.

    Args:
        data: Контекст пайплайна с ключами:
              'models_results', 'refinement_history', 'refined_diag', 'candidates', 'alpha'.
    """
    models_results = data['models_results']
    refinement_history = data['refinement_history']
    refined_diag = data['refined_diag']
    candidates = data['candidates']
    alpha = data.get('alpha', CONFIG["ALPHA"])

    def _generate() -> None:
        print("="*100)
        print("ОТЧЁТ 2: ИТЕРАЦИОННОЕ УЛУЧШЕНИЕ И REFINED МОДЕЛЬ")
        print("="*100)

        format_refinement_log(refinement_history)

        print("\n\n" + "="*80)
        print("[2] ДЕТАЛЬНЫЙ АНАЛИЗ REFINED МОДЕЛИ")
        print("="*80)

        if 'linear_refined' in models_results:
            ref_res = models_results['linear_refined']
            if 'metrics' in ref_res:
                print_metrics_template(ref_res['metrics'], "Линейная Регрессия (обработанная)")
            if 'anova_table' in ref_res:
                print(f"\n{'='*20} ANOVA TABLE {'='*20}")
                print(ref_res['anova_table'].to_string(index=False))

            print("\n--- STATSMODELS SUMMARY ---")
            print(ref_res['model_object'].summary())

            print("\n" + "-"*30 + " МАТЕМАТИЧЕСКОЕ УРАВНЕНИЕ " + "-"*30)
            print(models_results['linear_refined'].get('formula', 'Формула не найдена'))

            if 'Interpretation' in ref_res:
                print("\n" + "-"*30 + " ИНТЕРПРЕТАЦИЯ КОЭФФИЦИЕНТОВ " + "-"*30)
                print(ref_res['Interpretation'].sort_values(by='Коэффициент', ascending=False).to_string(index=False))

        print("\n\n[3] ТЕХНИЧЕСКАЯ ДИАГНОСТИКА (ОШИБКИ И ОСТАТКИ)")
        format_diagnostics_report(refined_diag, alpha)

        print("\n" + "="*100 + "\nКОНЕЦ ОТЧЁТА 2")

    _capture_output_and_save("text\\2_Очищенная_Линейная_Модель_Аналитика.txt", _generate)

def export_other_models_report(
    data: Dict[str, Any]
    ) -> None:
    """Генерирует Отчёт 3: Метрики и уравнения для всех альтернативных моделей.

    Перебирает стандартизированную, логарифмическую, полиномиальные,
    регуляризованные и ансамблевые модели. Для каждой выводит метрики,
    ANOVA, summary statsmodels, рейтинг важности признаков и математическое уравнение.

    Args:
        data: Контекст пайплайна с ключом 'models_results'.
    """
    models_results = data['models_results']

    def _generate() -> None:
        print("="*100)
        print("ОТЧЁТ 3: ОСТАЛЬНЫЕ МОДЕЛИ (Метрики и Уравнения)")
        print("="*100)

        model_keys = [
            ('standardized', 'СТАНДАРТИЗИРОВАННАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ'),
            ('log', 'ЛИНЕЙНАЯ РЕГРЕССИЯ ПО ЛОГАРИФМАМ'),
            ('poly2', 'ПОЛИНОМИАЛЬНАЯ РЕГРЕССИЯ (Степень 2)'),
            ('poly3', 'ПОЛИНОМИАЛЬНАЯ РЕГРЕССИЯ (Степень 3)'),
            ('ridge_poly2', 'RIDGE РЕГРЕССИЯ (Степень 2)'),
            ('lasso_poly2', 'LASSO РЕГРЕССИЯ (Степень 2)'),
            ('random_forest', 'RANDOM FOREST'),
            ('catboost', 'CATBOOST')
        ]

        for key, title in model_keys:
            if key not in models_results:
                print(f"\nМодель {title} отсутствует в результатах. Пропускаю.")
                continue

            res = models_results[key]
            print(f"\n\n{'='*40} {title} {'='*40}")

            # 1. Метрики
            print_metrics_template(res.get('metrics', res), title)
            model_obj = res.get('model_object')

            # 2. ANOVA
            if 'anova_table' in res:
                print(f"\n{'='*20} ANOVA TABLE {'='*20}")
                print(res['anova_table'].to_string(index=False))

            # 3. Summary
            if model_obj is not None and hasattr(model_obj, 'summary'):
                print(f"\n{'='*20} MODEL SUMMARY {'='*20}")
                print(model_obj.summary())

            if 'Interpretation' in res and not res['Interpretation'].empty:
                interp = res['Interpretation']
                potential_cols = ['Важность (%)', 'Важность', 'Влияние на цену (%)', 'Бета-коэффициент', 'Коэффициент']
                sort_col = next((c for c in potential_cols if c in interp.columns), None)

                if sort_col:
                    print(f"\n[Рейтинг значимости] ({sort_col}):")
                    is_abs = sort_col in ['Коэффициент', 'Влияние на цену (%)', 'Бета-коэффициент']
                    print(interp.sort_values(by=sort_col, key=abs if is_abs else None, ascending=False).head(20).to_string(index=False))
                else:
                    num_cols = interp.select_dtypes(include=[np.number]).columns
                    if len(num_cols) > 0:
                        print(f"\n[Рейтинг] (по {num_cols[0]}):")
                        print(interp.sort_values(by=num_cols[0], ascending=False).head(20).to_string(index=False))

            # 4. Математические уравнения
            if 'formula' in res and res['formula']:
                print("\n" + "="*20 + " МАТЕМАТИЧЕСКОЕ УРАВНЕНИЕ " + "="*20)
                print(res['formula'])
            elif key == 'log':
                x_sm = res.get('x_train_sm')
                feat_names = x_sm.columns.tolist() if hasattr(x_sm, 'columns') else []
                print_log_model_equation(model_obj, feat_names)

            # 5. Регуляризация
            if 'nnz_msg' in res:
                print(f"\n{res['nnz_msg']}")

        print("\n" + "="*100 + "\nКОНЕЦ ОТЧЁТА 3")

    _capture_output_and_save("text\\3_Остальные_модели.txt", _generate)

def export_comprehensive_comparison(
    data: Dict[str, Any]
    ) -> None:
    """Генерирует Отчёт 4: Итоговая сводная таблица сравнения всех моделей.

    Вызывает функцию агрегации метрик и сохраняет результат в единый текстовый файл
    для быстрого сопоставления архитектур, гиперпараметров и статистических показателей.

    Args:
        data: Контекст пайплайна с ключами:
              'models_results', 'x_train_dict', 'alpha'.
    """
    models_results = data['models_results']
    x_train_dict = data.get('x_train_dict')
    alpha = data.get('alpha', CONFIG["ALPHA"])

    def _generate() -> None:
        display_comprehensive_models_comparison({
            'models_results': models_results,
            'x_train_dict': x_train_dict,
            'alpha': alpha
        })
        print("\n" + "="*100 + "\nКОНЕЦ ОТЧЁТА 4")

    _capture_output_and_save("text\\4_Сравнение_моделей.txt", _generate)


# ==============================================================================
# 9. СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ
# ==============================================================================

METRICS_QUALITY = ['R²', 'Adj. R²', 'R² (log)', 'Adj.R² (log)']
METRICS_ERROR = ['MAE', 'RMSE', 'MAPE (%)', 'Huber']
METRICS_STATS = ['AIC', 'BIC', 'LogLik']

def plot_models_metrics_comparison(
    data: Dict[str, Any], 
    save: bool = True
) -> Optional[Dict[str, str]]:
    """Строит групповые столбчатые диаграммы для сравнения метрик моделей.

    Визуализирует ключевые показатели качества (R², MAE, RMSE, AIC и др.) 
    по всем обученным моделям. Каждая группа метрик выводится в отдельном 
    файле для удобного сопоставления архитектур и гиперпараметров.

    Args:
        data: Словарь-контекст, содержащий ключи:
              'df_comparison' (DataFrame с агрегированными метриками моделей),
              'save' (флаг сохранения, переопределяет аргумент функции).
        save: Флаг сохранения графиков в файловую систему.

    Returns:
        Словарь с путями к сохранённым файлам или None при ошибке создания директории.
    """
    df_comparison = data['df_comparison']
    save_flag = data.get('save', save)

    viz_path = os.path.join(CONFIG['DATA_PATH_OUTPUT'], "visualizations", "metrics_compare")
    saved_files = {}

    if save_flag:
        try:
            os.makedirs(viz_path, exist_ok=True)
        except Exception as e:
            print(f"Ошибка создания папки для графиков: {e}")
            return None

    df_plot = df_comparison.copy().set_index('Модель')
    
    groups = {
        "quality": (['R²', 'Adj. R²', 'R² (log)', 'Adj.R² (log)'], "Метрики качества (R²)"),
        "error_absolute": (['MAE', 'RMSE'], "Абсолютные ошибки (MAE, RMSE)"),
        "error_relative": (['MAPE (%)'], "Относительная ошибка (MAPE %)"),
        "huber": (['Huber'], "Метрика Huber Loss"),
        "stats": (['AIC', 'BIC', 'LogLik'], "Статистические критерии (AIC, BIC)")
    }

    for key, (metrics, title) in groups.items():
        available_metrics = [m for m in metrics if m in df_plot.columns]
        subset = df_plot[available_metrics].apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')

        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        subset.T.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Значение')
        ax.set_xticklabels(subset.columns, rotation=0) 
        ax.legend(title='Модели', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save_flag and viz_path:
            save_name = os.path.join(viz_path, f"comparison_{key}.png")
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            saved_files[key] = save_name
            print(f"График сохранен: {save_name}")
        
        plt.close(fig)

    return saved_files if saved_files else None

def plot_predictions_vs_actual(
    data: Dict[str, Any], 
    model_name: str = "Model", 
    save: bool = True
) -> Optional[str]:
    """Строит scatter plot: Фактические значения против предсказанных.
    
    Визуализирует точность прогноза модели. Добавляет диагональ y=x для
    визуальной оценки отклонений. В заголовок выносятся ключевые метрики
    (MAE, RMSE, R²) для быстрой интерпретации качества.
    
    Args:
        data: Словарь-контекст с ключами:
              'y_test' (фактические значения),
              'predictions' (предсказания модели),
              'save' (флаг сохранения).
        model_name: Идентификатор модели для заголовка и имени файла.
        save: Флаг сохранения графика.
    
    Returns:
        Путь к сохранённому файлу или None.
    """
    y_test = data['y_test']
    predictions = data['predictions']
    save_flag = data.get('save', save)
    
    viz_path = os.path.join(CONFIG['DATA_PATH_OUTPUT'], "visualizations", "predictions_vs_actual")
    
    if save_flag:
        try:
            os.makedirs(viz_path, exist_ok=True)
        except Exception as e:
            print(f"Ошибка создания папки для графиков: {e}")
            return None

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(y_test, predictions, alpha=0.6, s=30, 
               color='steelblue', edgecolors='black', linewidth=0.5,
               label=f'Прогнозы (MAE={mae:,.0f}, RMSE={rmse:,.0f})')
    
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Идеальный прогноз (y=x)')
    
    ax.set_title(f'Прогноз против фактических значений: {model_name}\nR² = {r2:.4f}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Фактическая цена (руб.)', fontsize=12)
    ax.set_ylabel('Предсказанная цена (руб.)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    formatter = ticker.StrMethodFormatter('{x:,.0f}')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    limit_max = y_test.max() * 1.1 
    ax.set_xlim(0, limit_max)
    ax.set_ylim(0, limit_max)
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    if save_flag and viz_path:
        safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        save_name = os.path.join(viz_path, f"pred_vs_actual_{safe_name}.png")
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"График сохранен: {save_name}")
    
    plt.close(fig)
    return save_name

def plot_residuals_analysis(
    data: Dict[str, Any], 
    model_name: str = "Model", 
    save: bool = True
) -> Optional[Dict[str, Union[float, str]]]:
    """Строит комплексный анализ распределения остатков модели.
    
    Включает два подграфика:
    1. Гистограмма с оценкой плотности (KDE) и теоретическим нормальным
       распределением для визуальной оценки формы.
    2. Q-Q plot для проверки гипотезы о нормальности остатков.
    Добавляет аннотации с тестом Шапиро-Уилка, асимметрией и эксцессом.
    
    Args:
        data: Словарь-контекст с ключами:
              'residuals' (остатки модели) ИЛИ
              'y_true' и 'y_pred' (для расчёта остатков),
              'save' (флаг сохранения).
        model_name: Идентификатор модели для заголовка.
        save: Флаг сохранения графика.
    
    Returns:
        Словарь со статистиками (skewness, kurtosis, shapiro_p и др.)
        или None при ошибке.
    """
    residuals = data.get('residuals')
    y_true = data.get('y_true')
    y_pred = data.get('y_pred')
    save_flag = data.get('save', save)
    
    if residuals is None and y_true is not None and y_pred is not None:
        residuals = np.array(y_true) - np.array(y_pred)
    
    if residuals is None or len(residuals) == 0:
        print(f"[Warning] Нет данных об остатках для модели '{model_name}'")
        return None

    residuals = np.asarray(residuals).flatten()
    
    skew_val = stats.skew(residuals)
    kurt_val = stats.kurtosis(residuals, fisher=False)
    kurt_fisher = stats.kurtosis(residuals, fisher=True)
    shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000]) if len(residuals) > 5000 else stats.shapiro(residuals)
    
    normality_status = "Нормально" if shapiro_p > CONFIG['ALPHA'] else "Не нормально"
    skew_status = "Симметрично" if abs(skew_val) < 0.5 else f"{'Право' if skew_val > 0 else 'Лево'}скошено"
    kurt_status = "Норм. острота" if abs(kurt_fisher) < 1 else f"{'Островершинное' if kurt_fisher > 0 else 'Плосковершинное'}"

    viz_path = os.path.join(CONFIG['DATA_PATH_OUTPUT'], "visualizations", "residuals_analysis")
    
    if save_flag:
        try:
            os.makedirs(viz_path, exist_ok=True)
        except Exception as e:
            print(f"Ошибка создания папки для графиков: {e}")
            return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1 = axes[0]
    ax1.hist(residuals, bins=50, density=True, alpha=0.6, color='steelblue', 
             edgecolor='black', linewidth=0.5, label='Остатки')
    
    if len(residuals) > 2:
        kde = gaussian_kde(residuals)
        x_range = np.linspace(residuals.min(), residuals.max(), 200)
        ax1.plot(x_range, kde(x_range), color='darkred', linewidth=2, label='KDE')
    
    mu, std = np.mean(residuals), np.std(residuals)
    x_norm = np.linspace(residuals.min(), residuals.max(), 200)
    p_norm = stats.norm.pdf(x_norm, mu, std)
    ax1.plot(x_norm, p_norm, 'g--', linewidth=1.5, label='Нормальное распред.')
    
    ax1.set_title('Распределение остатков', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel('Значение остатка', fontsize=11)
    ax1.set_ylabel('Плотность', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    
    stats_text = (f"Skew: {skew_val:.3f} ({skew_status})\n"
                  f"Kurt: {kurt_val:.3f} ({kurt_status})\n"
                  f"Mean: {np.mean(residuals):.2e}\n"
                  f"Std: {np.std(residuals):.2e}")
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax2 = axes[1]
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (проверка нормальности)', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    shapiro_text = (f"Shapiro-Wilk test:\n"
                    f"p-value = {shapiro_p:.4e}\n"
                    f"α = {CONFIG['ALPHA']} → {normality_status}")
    ax2.text(0.02, 0.98, shapiro_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    fig.suptitle(f'Анализ остатков: {model_name}', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_flag and viz_path:
        safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
        save_name = os.path.join(viz_path, f"residuals_analysis_{safe_name}.png")
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"График анализа остатков сохранен: {save_name}")
    
    plt.close(fig)
    
    return {
        'skewness': skew_val,
        'kurtosis': kurt_val,
        'kurtosis_fisher': kurt_fisher,
        'shapiro_p': shapiro_p,
        'normality_status': normality_status,
        'mean': np.mean(residuals),
        'std': np.std(residuals)
    }

def plot_residuals_vs_fitted(
    data: Dict[str, Any], 
    model_name: str = "Model", 
    save: bool = True
) -> Optional[Dict[str, Union[float, str]]]:
    """Строит диагностику остатков против предсказанных значений.
    
    Помогает выявить гетероскедастичность (непостоянство дисперсии ошибок) 
    и нелинейность, неучтённую моделью. Включает:
    1. Scatter plot остатков с линией скользящего среднего для выявления трендов.
    2. Бар-чарт статистик остатков по квантилям предсказаний.
    Рассчитывает отношение стандартных отклонений в двух группах для оценки 
    гомоскедастичности.
    
    Args:
        data: Словарь-контекст с ключами:
              'fitted_values' ИЛИ 'y_pred' (предсказания),
              'residuals' ИЛИ ('y_true' и 'y_pred' для расчёта),
              'save' (флаг сохранения).
        model_name: Идентификатор модели для заголовка.
        save: Флаг сохранения графика.
    
    Returns:
        Словарь со статистиками (resid_mean, hetero_ratio и др.) 
        или None при ошибке.
    """
    fitted_values = data.get('fitted_values')
    if fitted_values is None:
        fitted_values = data.get('y_pred')
        
    residuals = data.get('residuals')
    y_true = data.get('y_true')
    y_pred = data.get('y_pred')
    save_flag = data.get('save', save)

    if residuals is None and y_true is not None and y_pred is not None:
        residuals = np.array(y_true) - np.array(y_pred)
        
    if fitted_values is None or residuals is None:
        print(f"[Warning] Нет данных для графика 'Остатки против предсказанных' для модели '{model_name}'")
        return None

    fitted_values = np.asarray(fitted_values).flatten()
    residuals = np.asarray(residuals).flatten()
    
    if len(fitted_values) != len(residuals):
        print(f"[Error] Длина fitted_values ({len(fitted_values)}) != длины residuals ({len(residuals)})")
        return None

    resid_mean = np.mean(residuals)
    resid_std = np.std(residuals)
    resid_min = np.min(residuals)
    resid_max = np.max(residuals)
    
    median_fit = np.median(fitted_values)
    group1_std = np.std(residuals[fitted_values <= median_fit])
    group2_std = np.std(residuals[fitted_values > median_fit])
    hetero_ratio = max(group1_std, group2_std) / min(group1_std, group2_std) if min(group1_std, group2_std) > 0 else 0
    hetero_status = "Возможна гетероскедастичность" if hetero_ratio > 1.5 else "Гомоскедастичность (OK)"

    viz_path = os.path.join(CONFIG['DATA_PATH_OUTPUT'], "visualizations", "residuals_vs_fitted")
    
    if save_flag:
        try:
            os.makedirs(viz_path, exist_ok=True)
        except Exception as e:
            print(f"Ошибка создания папки для графиков: {e}")
            return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1 = axes[0]
    ax1.scatter(fitted_values, residuals, alpha=0.5, s=20, 
                color='steelblue', edgecolors='black', linewidth=0.3,
                label=f'Остатки (n={len(residuals)})')
    
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='y=0 (ноль ошибок)')
    
    if len(fitted_values) > 10:
        sorted_idx = np.argsort(fitted_values)
        fitted_sorted = fitted_values[sorted_idx]
        residuals_sorted = residuals[sorted_idx]
        window = max(10, min(100, int(len(fitted_values) * 0.1)))
        moving_avg = pd.Series(residuals_sorted).rolling(window=window, center=True).mean()
        ax1.plot(fitted_sorted, moving_avg, color='darkorange', linewidth=2.5, 
                 label=f'Скользящее среднее (window={window})')
    
    ax1.set_title('Остатки против предсказанных значений', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel('Предсказанные значения', fontsize=11)
    ax1.set_ylabel('Остатки', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    stats_text = (f"Mean: {resid_mean:.2e}\n"
                  f"Std: {resid_std:.2e}\n"
                  f"Min: {resid_min:.2e}\n"
                  f"Max: {resid_max:.2e}\n"
                  f"Range: {resid_max - resid_min:.2e}")
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax2 = axes[1]
    n_quantiles = 5
    quantiles = np.linspace(0, 1, n_quantiles + 1)
    fitted_quantiles = np.quantile(fitted_values, quantiles)
    
    quantile_means = []
    quantile_stds = []
    quantile_labels = []
    
    for i in range(n_quantiles):
        lower = fitted_quantiles[i]
        upper = fitted_quantiles[i + 1]
        if i == n_quantiles - 1:
            mask = (fitted_values >= lower) & (fitted_values <= upper)
        else:
            mask = (fitted_values >= lower) & (fitted_values < upper)
        
        if np.sum(mask) > 0:
            resid_in_quantile = residuals[mask]
            quantile_means.append(np.mean(resid_in_quantile))
            quantile_stds.append(np.std(resid_in_quantile))
            quantile_labels.append(f"Q{i+1}\n({lower:.0f}-{upper:.0f})")
    
    x_pos = np.arange(len(quantile_means))
    ax2.bar(x_pos, quantile_means, yerr=quantile_stds, capsize=5,
            color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5,
            label='Среднее ± Std')
    
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    
    ax2.set_title('Статистика остатков по квантилям предсказаний', fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel('Квантили предсказанных значений', fontsize=11)
    ax2.set_ylabel('Среднее остатков', fontsize=11)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(quantile_labels, fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    
    hetero_text = (f"Ratio std: {hetero_ratio:.2f}\n"
                   f"{hetero_status}\n"
                   f"Group1 std: {group1_std:.2e}\n"
                   f"Group2 std: {group2_std:.2e}")
    ax2.text(0.02, 0.98, hetero_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', 
                                               facecolor='lightgreen' if hetero_ratio <= 1.5 else 'salmon', 
                                               alpha=0.3))
    
    fig.suptitle(f'Диагностика остатков: {model_name}', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_flag and viz_path:
        safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
        save_name = os.path.join(viz_path, f"residuals_vs_fitted_{safe_name}.png")
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"График сохранен: {save_name}")
    
    plt.close(fig)
    
    return {
        'resid_mean': resid_mean,
        'resid_std': resid_std,
        'hetero_ratio': hetero_ratio,
        'hetero_status': hetero_status,
        'group1_std': group1_std,
        'group2_std': group2_std
    }

def plot_correlation_heatmap(
    data: Dict[str, Any], 
    target_col: str = CONFIG['TARGET_COL'], 
    corr_threshold: float = 0.7,
    highlight_target: bool = True, 
    save: bool = True
) -> Optional[Dict[str, Any]]:
    """Строит тепловую карту корреляционной матрицы с выделением связей с целевой переменной.

    Визуализирует линейные зависимости между признаками с использованием дивергентной 
    цветовой схемы. При включённом флаге `highlight_target` выделяет строку и столбец 
    таргета золотой рамкой, сортирует признаки по силе связи с целевой переменной 
    и добавляет статистику сильных корреляций в угол графика.

    Args:
        data: Словарь-контекст, содержащий ключи:
              'corr_matrix' (DataFrame корреляционной матрицы),
              'model_name' (опционально: имя модели для заголовка),
              'save' (флаг сохранения, переопределяет аргумент функции).
        target_col: Имя целевой переменной для выделения и сортировки.
        corr_threshold: Порог абсолютного значения корреляции для выделения сильных связей.
        highlight_target: Флаг выделения целевой переменной на графике.
        save: Флаг сохранения графика в файловую систему.

    Returns:
        Словарь со статистикой матрицы (количество признаков, сильных пар, 
        максимальная абсолютная корреляция) или None при ошибке/пустых данных.
    """
    corr_matrix = data['corr_matrix']
    save_flag = data.get('save', save)
    
    if corr_matrix is None or corr_matrix.empty:
        print("[Warning] Пустая матрица корреляций для визуализации")
        return None

    viz_path = os.path.join(CONFIG['DATA_PATH_OUTPUT'], "visualizations", "correlation_heatmap")
    
    if save_flag:
        try:
            os.makedirs(viz_path, exist_ok=True)
        except Exception as e:
            print(f"Ошибка создания папки для графиков: {e}")
            return None

    df_plot = corr_matrix.copy()
    
    if target_col in df_plot.columns and highlight_target:
        target_corr = df_plot[target_col].drop(target_col).abs().sort_values(ascending=False)
        ordered_cols = [target_col] + list(target_corr.index)
        df_plot = df_plot.loc[ordered_cols, ordered_cols]
    
    n_features = len(df_plot)
    figsize = (max(12, min(20, n_features * 0.6)), max(10, min(20, n_features * 0.6)))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sns.heatmap(
        df_plot,
        ax=ax,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'shrink': 0.8, 'label': 'Коэффициент корреляции'},
        vmin=-1, vmax=1
    )
    
    if highlight_target and target_col in df_plot.index:
        target_idx = df_plot.index.get_loc(target_col)
        ax.axvline(x=target_idx, color='gold', linewidth=3, linestyle='-', 
                   label=f'Целевая: {target_col}', zorder=10)
        ax.axhline(y=target_idx, color='gold', linewidth=3, linestyle='-', zorder=10)
        legend_elements = [Patch(facecolor='gold', edgecolor='black', label=f'> {target_col}')]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True)
    
    strong_pairs = []
    cols = df_plot.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = df_plot.iloc[i, j]
            if abs(val) > corr_threshold and not np.isnan(val):
                strong_pairs.append((cols[i], cols[j], val))
    
    max_corr = df_plot.abs().where(~np.eye(n_features, dtype=bool), np.nan).max().max()
    stats_text = (f"Всего признаков: {n_features}\n"
                  f"Сильных пар (|r|>{corr_threshold}): {len(strong_pairs)}\n"
                  f"Макс. |r|: {max_corr:.3f}")
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4),
            zorder=100)
    
    title_suffix = f" ({data.get('model_name')})" if data.get('model_name') else ""
    ax.set_title(f'Корреляционная матрица{title_suffix}', 
                 fontsize=15, fontweight='bold', pad=20)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    if save_flag and viz_path:
        model_name = data.get('model_name')
        safe_name = (model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_') 
                     if model_name else 'full_dataset')
        save_name = os.path.join(viz_path, f"correlation_heatmap_{safe_name}.png")
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Heatmap корреляций сохранен: {save_name}")
    
    plt.close(fig)
    
    return {
        'n_features': n_features,
        'strong_pairs_count': len(strong_pairs),
        'strong_pairs_list': strong_pairs,
        'max_abs_corr': max_corr
    }

def plot_pairwise_vs_partial_correlation(
    data: Dict[str, Any], 
    target_col: str = 'price', 
    top_n: int = 20, 
    delta_threshold: float = 0.3,
    save: bool = True
) -> None:
    """Строит сгруппированную горизонтальную гистограмму сравнения парной и частной корреляций.

    Для каждого признака отображает два столбца: парную корреляцию с таргетом 
    и частную корреляцию (контролируя влияние остальных признаков). Признаки 
    с расхождением выше порога выделяются красным цветом для визуального 
    обнаружения искажений, вызванных мультиколлинеарностью.

    Args:
        data: Словарь-контекст, содержащий ключи:
              'corr_extended' (DataFrame с колонками 'Признак', 'Парная', 'Частная', 'Дельта'),
              'model_name' (идентификатор модели для заголовка),
              'save' (флаг сохранения).
        target_col: Имя целевой переменной (для совместимости интерфейса).
        top_n: Количество признаков с наибольшей абсолютной парной корреляцией для отображения.
        delta_threshold: Порог расхождения между парной и частной корреляцией для выделения.
        save: Флаг сохранения графика.
    """
    corr_extended = data['corr_extended']
    model_name = data.get('model_name', 'Model')
    save_flag = data.get('save', save)
    
    if corr_extended is None or corr_extended.empty:
        return

    viz_path = os.path.join(CONFIG['DATA_PATH_OUTPUT'], "visualizations", "pairwise_vs_partial")
    if save_flag:
        os.makedirs(viz_path, exist_ok=True)

    df_plot = corr_extended.copy()
    df_plot['Парная_abs'] = df_plot['Парная'].abs()
    df_plot = df_plot.sort_values('Парная_abs', ascending=True).tail(top_n)
    
    ind = np.arange(len(df_plot)) 
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, max(8, len(df_plot) * 0.6)))

    rects1 = ax.barh(ind + width/2, df_plot['Парная'], width, 
                     label='Парная корреляция', color='steelblue', edgecolor='black', alpha=0.8)
    rects2 = ax.barh(ind - width/2, df_plot['Частная'], width, 
                     label='Частная корреляция', color='lightseagreen', edgecolor='black', alpha=0.8)

    ax.set_yticks(ind)
    ax.set_yticklabels(df_plot['Признак'])
    
    for i, tick in enumerate(ax.get_yticklabels()):
        if abs(df_plot.iloc[i]['Дельта']) > delta_threshold:
            tick.set_color('crimson')
            tick.set_weight('bold')

    def autolabel(rects):
        for rect in rects:
            width_val = rect.get_width()
            ax.annotate(f'{width_val:+.3f}',
                        xy=(width_val, rect.get_y() + rect.get_height() / 2),
                        xytext=(3 if width_val >= 0 else -3, 0),
                        textcoords="offset points",
                        ha='left' if width_val >= 0 else 'right', va='center', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f'Сравнение корреляций: {model_name}\n(Красным выделены искажения > {delta_threshold})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Коэффициент корреляции')
    ax.legend(loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()

    if save_flag:
        safe_name = model_name.lower().replace(' ', '_')
        plt.savefig(os.path.join(viz_path, f"grouped_corr_{safe_name}.png"), dpi=300, bbox_inches='tight')
    
    plt.close()

def create_tkinter_prediction_ui(
    data: Dict[str, Any], 
    target_col: str = CONFIG['TARGET_COL']
) -> None:
    models_results = data['models_results']
    x_train_original = data['x_train_original']
    final_data = data['final_data']
    
    root = tk.Tk()
    root.title("Прогноз стоимости квартиры")
    root.geometry("650x800")

    forbidden_keywords = ['region_of_moscow_', 'region_original', target_col]
    
    numeric_features = [
        col for col in x_train_original.columns 
        if x_train_original[col].dtype in ['int64', 'float64'] 
        and not any(word in col for word in forbidden_keywords)
        and col not in ['is_apartments', 'is_new']
    ]
    
    region_cols = [col for col in x_train_original.columns if col.startswith('region_of_moscow_')]
    region_names = [col.replace('region_of_moscow_', '').replace('_', ' ') for col in region_cols]

    # Определяем базовый (референтный) регион - тот, который был удален при One-Hot Encoding
    # Обычно это region_of_moscow_NAR (Северный административный округ)
    all_possible_regions = [
        'CAR', 'EAR', 'EAR', 'NEAR', 'NWAR', 'SAR', 'SEAR', 'SWAR', 'WAR', 'NAR'
    ]
    # Находим какой регион отсутствует в region_cols - это и есть базовый
    base_region_code = None
    for code in all_possible_regions:
        if f'region_of_moscow_{code}' not in region_cols:
            base_region_code = code
            break
    
    # Если не нашли отсутствующий регион, берем первый по умолчанию
    if base_region_code is None:
        base_region_code = 'NAR'  # Северный округ как стандартный базовый
    
    BASE_REGION_TEXT = f"По умолчанию ({base_region_code})"
    
    # Создаем список регионов с опцией "По умолчанию"
    region_options = [BASE_REGION_TEXT] + sorted(region_names)

    main_container = ttk.Frame(root)
    main_container.pack(fill="both", expand=True, padx=10, pady=10)

    canvas = tk.Canvas(main_container)
    scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    entries = {}
    ttk.Label(scrollable_frame, text="ОСНОВНЫЕ ПАРАМЕТРЫ (в скобках диапазон обучения)", font=('Arial', 10, 'bold')).pack(pady=5)
    
    for feat in numeric_features:
        frame = ttk.Frame(scrollable_frame)
        frame.pack(fill="x", pady=2)
        
        min_v = x_train_original[feat].min()
        max_v = x_train_original[feat].max()
        default_val = x_train_original[feat].median()
        
        label_text = f"{feat} ({min_v:.0f}-{max_v:.0f}):"
        
        if any(word in feat for word in ['floor', 'rooms', 'year', 'age']):
            formatted_val = f"{int(default_val)}"
        else:
            formatted_val = f"{default_val:.2f}"

        ttk.Label(frame, text=label_text, width=35).pack(side="left")
        entry = ttk.Entry(frame)
        entry.insert(0, formatted_val)
        entry.pack(side="right", expand=True, fill="x")
        entries[feat] = entry

    ttk.Label(scrollable_frame, text="\nТИП НЕДВИЖИМОСТИ", font=('Arial', 9, 'bold')).pack(pady=(10, 5))
    is_apartments_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(scrollable_frame, text="is_apartments (апартаменты)", variable=is_apartments_var).pack(anchor="w")
    
    is_new_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(scrollable_frame, text="is_new (новостройка)", variable=is_new_var).pack(anchor="w")

    ttk.Label(scrollable_frame, text="\nЛОКАЦИЯ", font=('Arial', 10, 'bold')).pack(pady=5)

    region_var = tk.StringVar(value=BASE_REGION_TEXT)
    region_combo = ttk.Combobox(scrollable_frame, textvariable=region_var, values=region_options, state="readonly")
    region_combo.pack(fill="x", pady=5)

    ttk.Label(scrollable_frame, text="\nМОДЕЛЬ", font=('Arial', 10, 'bold')).pack(pady=5)
    
    excluded_keys = ['refined_x_train', 'refined_x_test', 'refined_x_train_initial', 'refined_x_test_initial']
    model_display_options = []
    model_key_map = {}

    for key in models_results.keys():
        if key not in excluded_keys:
            res = models_results[key]
            r2 = res.get('R2') or res.get('metrics', {}).get('R2', 0)
            rmse = res.get('RMSE') or res.get('metrics', {}).get('RMSE', 0)
            display_str = f"{key} | R²:{r2:.3f} | RMSE:{rmse:,.0f}"
            model_display_options.append(display_str)
            model_key_map[display_str] = key

    default_model_str = next((s for s in model_display_options if "random_forest" in s.lower()), model_display_options[0])
    model_var = tk.StringVar(value=default_model_str)
    model_combo = ttk.Combobox(scrollable_frame, textvariable=model_var, values=model_display_options, state="readonly")
    model_combo.pack(fill="x", pady=5)


    def predict():
        try:
            input_dict = {}
            warnings = []


            for feat in numeric_features:
                val = float(entries[feat].get())
                min_v = x_train_original[feat].min()
                max_v = x_train_original[feat].max()
                
                if val < min_v or val > max_v:
                    warnings.append(f"- {feat}: введено {val}, диапазон [{min_v:.1f} - {max_v:.1f}]")
                
                input_dict[feat] = int(round(val)) if any(w in feat for w in ['floor', 'rooms', 'year', 'age']) else val
            
            # Показ окна предупреждения
            if warnings:
                warn_msg = "Значения выходят за границы обучения:\n\n" + "\n".join(warnings) + "\n\nПродолжить расчет?"
                if not messagebox.askyesno("Предупреждение", warn_msg):
                    return

            input_dict['is_apartments'] = 1 if is_apartments_var.get() else 0
            input_dict['is_new'] = 1 if is_new_var.get() else 0
            
            selected_reg = region_var.get()
            selected_display_str = model_var.get()
            model_key = model_key_map[selected_display_str]
            

            use_base_region = False
            if model_key == 'linear_refined':

                refined_regions = [col for col in models_results['linear_refined']['x_train_sm'].columns 
                                  if col.startswith('region_of_moscow_')]
                refined_region_names = [col.replace('region_of_moscow_', '').replace('_', ' ') 
                                       for col in refined_regions]
                
                if selected_reg != BASE_REGION_TEXT and selected_reg not in refined_region_names:
                    use_base_region = True
                    messagebox.showinfo(
                        "Информация", 
                        f"Выбранный регион '{selected_reg}' отсутствует в очищенной модели.\n"
                        f"Автоматически используется базовый регион ({base_region_code})."
                    )

            # Устанавливаем дамми-переменные для регионов
            for col in region_cols:
                if selected_reg == BASE_REGION_TEXT or use_base_region:
                    # Если выбран базовый регион или используется замена — зануляем все дамми
                    input_dict[col] = 0
                else:
                    # Если выбран конкретный регион — активируем только его
                    clean_name = col.replace('region_of_moscow_', '').replace('_', ' ')
                    input_dict[col] = 1 if clean_name == selected_reg else 0
            
            model_res = models_results[model_key]
            model_obj = model_res['model_object']
            
            input_df = pd.DataFrame([input_dict])
            
            # Логика расчета предсказания (statsmodels / sklearn / ensembles)
            if model_key in ['linear', 'linear_robust', 'linear_refined', 'standardized', 'log']:
                train_cols = model_res['x_train_sm'].columns.tolist()
                if 'const' in train_cols: train_cols.remove('const')
                
                current_input = input_df[train_cols].copy()
                if model_key == 'standardized' and 'scaler' in model_res:
                    current_input = pd.DataFrame(model_res['scaler'].transform(current_input), columns=current_input.columns)

                input_sm = sm.add_constant(current_input, has_constant='add', prepend=True)
                input_sm = input_sm[model_res['x_train_sm'].columns]
                
                pred_obj = model_obj.get_prediction(input_sm)
                prediction = pred_obj.predicted_mean[0]
                ci = pred_obj.conf_int(obs=True, alpha=CONFIG['ALPHA'])
                lower, upper = ci[0][0], ci[0][1]
                
                if model_key == 'log':
                    prediction, lower, upper = np.expm1(prediction), np.expm1(lower), np.expm1(upper)
            
            elif 'poly' in model_key:
                poly_transformer = model_res['poly_transformer']
                base_features = model_res.get('feature_names', numeric_features + ['is_apartments', 'is_new'] + region_cols)
                current_input = input_df[base_features]
                input_poly = poly_transformer.transform(current_input)
                if 'scaler' in model_res:
                    input_poly = model_res['scaler'].transform(input_poly)
                prediction = model_obj.predict(input_poly)[0]
                rmse = model_res.get('RMSE') or model_res.get('metrics', {}).get('RMSE', 0)
                z = stats.norm.ppf(1 - CONFIG['ALPHA'] / 2)
                lower, upper = prediction - z * rmse, prediction + z * rmse
            else:
                train_cols = x_train_original.columns.tolist()
                prediction = model_obj.predict(input_df[train_cols])[0]
                rmse = model_res.get('RMSE') or model_res.get('metrics', {}).get('RMSE', 0)
                z = stats.norm.ppf(1 - CONFIG['ALPHA'] / 2)
                lower, upper = prediction - z * rmse, prediction + z * rmse

            # Формируем сообщение о результате с информацией о регионе
            actual_region = base_region_code if (selected_reg == BASE_REGION_TEXT or use_base_region) else selected_reg
            res_message = (
                f"Модель: {model_key}\n"
                f"Регион: {actual_region}\n"
                f"Прогноз: {max(0, prediction):,.0f} руб.\n"
                f"{int((1-CONFIG['ALPHA'])*100)}% ДИ: [{max(0, lower):,.0f} - {upper:,.0f}] руб."
            )
            messagebox.showinfo("Результат", res_message)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Детали: {str(e)}")

    ttk.Button(scrollable_frame, text="РАССЧИТАТЬ СТОИМОСТЬ", command=predict).pack(pady=20)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    root.mainloop()
    
# ==============================================================================
# 10. ТОЧКА ВХОДА
# ==============================================================================

def main() -> None:
    """Основная функция-оркестратор пайплайна эконометрического анализа.
    
    Последовательно выполняет:
    1. Загрузку и предобработку данных (очистка, кодирование, train/test split).
    2. Обучение набора базовых моделей (линейные, полиномиальные, ансамбли).
    3. Сбор диагностических метрик и корреляционного анализа.
    4. Итеративный отбор признаков и обучение очищенной модели.
    5. Генерацию текстовых отчётов (4 файла) и визуализаций (графики).
    6. Запуск интерактивного интерфейса для ручного прогнозирования.
    
    Все промежуточные результаты передаются через единый словарь-контекст `state`,
    что обеспечивает согласованность данных и упрощает отладку. Ошибки обрабатываются
    с выводом трассировки стека для диагностики.
    """
    os.makedirs(CONFIG["DATA_PATH_INPUT"], exist_ok=True)
    os.makedirs(CONFIG["DATA_PATH_OUTPUT"], exist_ok=True)
    
    print("ЗАПУСК ПРОГРАММЫ АНАЛИЗА ДАННЫХ")
    
    state = {
        'models_results': {},
        'final_data': None,
        'x_train': None,
        'x_test': None,
        'y_train': None,
        'y_test': None,
        'diagnostics': {},
        'corr_data': None,
        'robustness_df': None,
        'selection_report': None,
        'candidates': [],
        'refinement_history': None,
        'refined_diag': None,
        'x_train_history': {}
    }
    
    try:
        # 1. Загрузка данных
        state['final_data'] = clean_and_prepare_data(load_and_init_data())
        
        # 2. Разделение на признаки/таргет и train/test
        state['x_train'], state['x_test'], state['y_train'], state['y_test'] = prepare_features(
            state['final_data'], del_cols=CONFIG["DROP_FEATURES"]
        )
        
        # 3. Построение базовых моделей
        state['models_results'] = build_baseline_models(
            data={
                'x_train': state['x_train'],
                'x_test': state['x_test'],
                'y_train': state['y_train'],
                'y_test': state['y_test']
            },
            silent=False
        )
        
        # 4. Обучение робастной модели (HC3) для сравнения
        print("\nОбучение робастной модели (HC3)...")
        state['models_results']['linear_robust'] = run_linear_regression(
            {'x_train': state['x_train'], 'x_test': state['x_test'],
             'y_train': state['y_train'], 'y_test': state['y_test']},
            cov_type='HC3', silent=True
        )
        
        # 5. Сбор диагностических данных
        print("\nСбор диагностических данных...")
        state['diagnostics']['baseline'] = collect_diagnostic_data(
            {'model_result': state['models_results']['linear'],
             'x_train': state['x_train'], 'y_train': state['y_train'],
             'x_test': state['x_test']},
            alpha=CONFIG["ALPHA"]
        )
        state['corr_data'] = collect_correlations(
            {'final_data': state['final_data']}, target_col=CONFIG["TARGET_COL"]
        )
        state['robustness_df'] = collect_robustness_comparison(
            {'res_raw': state['models_results']['linear'],
             'res_robust': state['models_results']['linear_robust']},
            alpha=CONFIG["ALPHA"]
        )
        
        # Консольный вывод для мониторинга
        print("\n--- КОНСОЛЬНЫЙ ВЫВОД ДИАГНОСТИКИ ---")
        format_diagnostics_report(state['diagnostics']['baseline'])
        format_correlations_report(state['corr_data'])
        format_robustness_report(state['robustness_df'])
        
        # 6. Селекция признаков и итеративное улучшение
        state['selection_report'], state['candidates'] = get_feature_selection_advice(
            {'model_result': state['models_results']['linear_robust'],
             'x_train': state['x_train'], 'final_data': state['final_data']},
            alpha=CONFIG["ALPHA"],
            priority_threshold=CONFIG["priority_threshold"],
            silent=False
        )
        
        refinement_result = run_iterative_refinement(
            {'x_train': state['x_train'], 'x_test': state['x_test'],
             'y_train': state['y_train'], 'y_test': state['y_test']},
            candidates_to_drop=state['candidates'], cov_type='HC3', silent=False
        )
        
        state['models_results']['linear_refined'] = refinement_result['last_model']
        state['models_results']['refined_x_train'] = refinement_result['refined_x_train']
        state['models_results']['refined_x_test'] = refinement_result['refined_x_test']
        state['refinement_history'] = refinement_result['history']
        
        print("\nСбор диагностики финальной модели...")
        state['refined_diag'] = collect_diagnostic_data(
            {'model_result': state['models_results']['linear_refined'],
             'x_train': state['models_results']['refined_x_train'],
             'y_train': state['y_train'],
             'x_test': state['models_results']['refined_x_test']},
            alpha=CONFIG["ALPHA"]
        )
        
        # История признаков для итоговой таблицы
        state['x_train_history'] = {
            'linear': state['x_train'],
            'linear_robust': state['x_train'],
            'linear_refined': state['models_results']['refined_x_train']
        }
        
        # 7. Генерация текстовых отчётов
        print("\nГенерация текстовых отчётов...")
        export_linear_analysis_report({
            'models_results': state['models_results'],
            'diagnostics': state['diagnostics']['baseline'],
            'corr_data': state['corr_data'],
            'robustness_df': state['robustness_df'],
            'selection_report': state['selection_report'],
            'candidates': state['candidates'],
            'alpha': CONFIG["ALPHA"]
        })
        export_refined_model_report({
            'models_results': state['models_results'],
            'refinement_history': state['refinement_history'],
            'refined_diag': state['refined_diag'],
            'candidates': state['candidates'],
            'alpha': CONFIG["ALPHA"]
        })
        export_other_models_report({'models_results': state['models_results']})
        export_comprehensive_comparison({
            'models_results': state['models_results'],
            'x_train_dict': state['x_train_history'],
            'alpha': CONFIG["ALPHA"]
        })
        
        # 8. Генерация визуализаций
        print("\nГенерация графиков...")


        models_to_plot = {
            'linear': 'Linear Regression',
            'standardized': 'Standardized Regression',
            'log': 'Logarithmic Regression',
            'poly2': 'Polynomial Regression (deg 2)',
            'poly3': 'Polynomial Regression (deg 3)',
            'ridge_poly2': 'Ridge Regression (Poly 2)',
            'lasso_poly2': 'Lasso Regression (Poly 2)',
            'random_forest': 'Random Forest',
            'catboost': 'CatBoost',
            'linear_robust': 'Linear Regression (Robust HC3)',
            'linear_refined': 'Linear Regression (Refined)',
        }


        for model_key, model_name in models_to_plot.items():
            if model_key not in state['models_results']:
                continue
                
            res = state['models_results'][model_key]
            
            plot_predictions_vs_actual(
                data={'y_test': state['y_test'], 'predictions': res['predictions']},
                model_name=model_name,
                save=True
            )
            

            if hasattr(res.get('model_object'), 'resid'):
                plot_residuals_analysis(
                    data={'residuals': res['model_object'].resid},
                    model_name=model_name,
                    save=True
                )
            elif 'predictions' in res:
                plot_residuals_analysis(
                    data={'y_true': state['y_test'], 'y_pred': res['predictions']},
                    model_name=model_name,
                    save=True
                )
            

            if hasattr(res.get('model_object'), 'fittedvalues'):
                plot_residuals_vs_fitted(
                    data={
                        'fitted_values': res['model_object'].fittedvalues,
                        'residuals': res['model_object'].resid
                    },
                    model_name=model_name,
                    save=True
                )
            elif 'predictions' in res:

                try:
                    train_preds = res['model_object'].predict(state['x_train'])
                    plot_residuals_vs_fitted(
                        data={
                            'fitted_values': train_preds,
                            'residuals': state['y_train'] - train_preds
                        },
                        model_name=model_name,
                        save=True
                    )
                except:
                    pass 

        print("\nГенерация Heatmap корреляций...")
        plot_correlation_heatmap({
            'corr_matrix': state['corr_data']['matrix'],
            'model_name': 'linear', 'save': True
        })
        if 'refined_x_train' in state['models_results']:
            refined_cols = state['models_results']['refined_x_train'].columns.tolist() + [CONFIG["TARGET_COL"]]
            refined_corr = state['final_data'][refined_cols].corr()
            plot_correlation_heatmap({
                'corr_matrix': refined_corr, 'model_name': 'linear_refined', 'save': True
            })
            
        print("\nГенерация графиков 'Парная против Частной корреляции'...")
        plot_pairwise_vs_partial_correlation({
            'corr_extended': state['corr_data']['extended'],
            'model_name': 'Linear_All_Features', 'save': True
        })
        if 'linear_refined' in state['models_results']:
            refined_features = state['models_results']['refined_x_train'].columns.tolist()
            refined_subset = state['final_data'][refined_features + [CONFIG["TARGET_COL"]]]
            corr_refined = collect_correlations({'final_data': refined_subset}, target_col=CONFIG["TARGET_COL"])
            plot_pairwise_vs_partial_correlation({
                'corr_extended': corr_refined['extended'],
                'model_name': 'Linear_Refined', 'save': True
            })
            

        print("\nЗапуск интерактивного модуля прогнозирования...")
        create_tkinter_prediction_ui({
            'models_results': state['models_results'],
            'x_train_original': state['x_train'],
            'final_data': state['final_data']
        })
        
    except Exception as e:
        print(f"\n[КРИТИЧЕСКАЯ ОШИБКА]: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()