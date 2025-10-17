# -*- coding: utf-8 -*-
"""
Módulo de Engenharia de Features.

Este módulo contém funções para criar variáveis preditoras (features)
e a variável-alvo (target) a partir dos dados brutos OHLCV.
"""

import numpy as np
import pandas as pd

# Importa a função de download para usar nos testes
from data_downloader import fetch_ohlcv_data

# --- Constantes ---
# Fator para anualizar a volatilidade de dados horários (mercado 24/7)
ANNUALIZATION_FACTOR = np.sqrt(365 * 24)

# Janelas de tempo (em horas) para cálculos de volatilidade, médias, etc.
SHORT_TERM_WINDOW = 24  # 1 dia
MID_TERM_WINDOW = 168  # 7 dias
LONG_TERM_WINDOW = 720 # 30 dias


def create_target_variable(df: pd.DataFrame, window: int = SHORT_TERM_WINDOW) -> pd.DataFrame:
    """
    Cria a variável-alvo: a volatilidade realizada das próximas 'window' horas.
    """
    print("Iniciando a criação da variável-alvo...")
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    future_volatility = df['log_returns'].shift(-window).rolling(window=window).std()
    df['target_volatility'] = future_volatility * ANNUALIZATION_FACTOR
    
    print("Criação da variável-alvo concluída.")
    return df

# --- Funções Modulares para Criação de Features ---

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features baseadas na volatilidade passada (realizada)."""
    print("Adicionando features de volatilidade...")
    
    # Volatilidade realizada para diferentes janelas de tempo passadas
    for window in [SHORT_TERM_WINDOW, MID_TERM_WINDOW, LONG_TERM_WINDOW]:
        col_name = f'vol_realizada_{window}h'
        df[col_name] = df['log_returns'].rolling(window=window).std() * ANNUALIZATION_FACTOR

    # Rácios de volatilidade para capturar aceleração
    df['vol_ratio_24_168'] = df[f'vol_realizada_{SHORT_TERM_WINDOW}h'] / df[f'vol_realizada_{MID_TERM_WINDOW}h']
    
    return df

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de momentum e retorno."""
    print("Adicionando features de momentum...")

    # Retornos acumulados para diferentes janelas
    for window in [SHORT_TERM_WINDOW, MID_TERM_WINDOW]:
        col_name = f'retorno_acumulado_{window}h'
        df[col_name] = df['log_returns'].rolling(window=window).sum()

    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_24h'] = true_range.rolling(window=SHORT_TERM_WINDOW).mean()
    
    return df

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features baseadas em volume de negociação."""
    print("Adicionando features de volume...")

    # Médias móveis de volume
    for window in [SHORT_TERM_WINDOW, MID_TERM_WINDOW]:
        col_name = f'media_movel_volume_{window}h'
        df[col_name] = df['volume'].rolling(window=window).mean()
        
    # Anomalia de volume (volume da última hora vs média das 24h)
    df['volume_ratio_1_24'] = df['volume'] / df[f'media_movel_volume_{SHORT_TERM_WINDOW}h']

    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features baseadas no tempo (cíclicas)."""
    print("Adicionando features de tempo...")
    
    df['hora_do_dia'] = df.index.hour
    df['dia_da_semana'] = df.index.dayofweek # Segunda=0, Domingo=6
    df['mes_do_ano'] = df.index.month
    
    return df

# --- Função Mestra ---

def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo para criar todas as features a partir dos dados brutos.

    Args:
        df (pd.DataFrame): DataFrame com dados OHLCV e a coluna 'log_returns'.

    Returns:
        pd.DataFrame: DataFrame com todas as features adicionadas.
    """
    print("\nIniciando pipeline de criação de features...")
    
    # Copia o DF para evitar o SettingWithCopyWarning
    df_featured = df.copy()
    
    # Executa cada módulo de criação de feature em sequência
    df_featured = add_volatility_features(df_featured)
    df_featured = add_momentum_features(df_featured)
    df_featured = add_volume_features(df_featured)
    df_featured = add_time_features(df_featured)
    
    print("Pipeline de criação de features concluído.")
    return df_featured


if __name__ == '__main__':
    # Bloco de teste para o pipeline completo de engenharia de features
    
    print("\n--- Iniciando Teste do Módulo de Engenharia de Features (Completo) ---")
    
    TEST_SYMBOL = 'BTC/USDT'
    # Usamos um período maior para que as janelas longas (30 dias) sejam calculadas
    TEST_START_DATE = '2024-01-01'
    TEST_TIMEFRAME = '1h'

    # 1. Baixa os dados brutos
    data_raw = fetch_ohlcv_data(
        symbol=TEST_SYMBOL,
        start_date=TEST_START_DATE,
        timeframe=TEST_TIMEFRAME
    )

    if not data_raw.empty:
        # 2. Cria a variável-alvo
        data_with_target = create_target_variable(data_raw)

        # 3. Cria todas as features preditoras
        final_df = create_all_features(data_with_target)

        # 4. Limpa os dados de valores nulos (gerados pelas janelas móveis)
        #    Este seria o dataset final, pronto para o treinamento
        final_df_cleaned = final_df.dropna()

        print("\n--- Verificação do DataFrame Final ---")
        print(f"Total de colunas criadas: {len(final_df.columns)}")
        print("Nomes das colunas:")
        print(final_df.columns.tolist())
        
        print("\nAmostra dos dados finais (sem NaNs):")
        print(final_df_cleaned.head())
        
        print("\n" + "-"*40)
        print(f"Shape original: {final_df.shape}")
        print(f"Shape após remover NaNs: {final_df_cleaned.shape}")
        print(f"Perda de dados para aquecimento das features: {final_df.shape[0] - final_df_cleaned.shape[0]} linhas")
