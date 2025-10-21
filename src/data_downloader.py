# -*- coding: utf-8 -*-
"""
Módulo de Coleta de Dados para Ativos Financeiros.

Este módulo é responsável por se conectar à API da Binance
para baixar dados históricos de velas (OHLCV) para um determinado
ativo financeiro.

Funções:
    fetch_ohlcv_data: Baixa dados OHLCV em um período específico,
                      gerenciando a paginação da API.
"""

import ccxt
import pandas as pd
from datetime import datetime

def fetch_ohlcv_data(symbol: str, start_date: str, timeframe: str = '1h') -> pd.DataFrame:
    """
    Busca dados históricos OHLCV da Binance.

    A função gerencia a paginação para buscar todos os dados desde a
    data de início até o presente momento em um único chamado.

    Args:
        symbol (str): O símbolo do ativo no formato da corretora (ex: 'BTC/USDT').
        start_date (str): A data de início no formato 'YYYY-MM-DD'.
        timeframe (str, optional): O intervalo de tempo das velas. 
                                   Padrão é '1h' (1 hora).

    Returns:
        pd.DataFrame: Um DataFrame do Pandas com os dados OHLCV,
                      indexado por timestamp. Retorna um DataFrame vazio
                      se nenhum dado for encontrado.
    """
    print("Inicializando o coletor de dados...")
    
    # 1. Inicializa a conexão com a exchange (Binance, neste caso)
    exchange = ccxt.binance()

    # exchange.session.verify = True

    # 2. Converte a data de início para o formato de timestamp em milissegundos
    #    que a API da CCXT espera.
    try:
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    except ValueError:
        print("Erro: Formato de data inválido. Use 'YYYY-MM-DD'.")
        return pd.DataFrame()

    all_data = []
    
    print(f"Iniciando download dos dados para {symbol} desde {start_date}...")

    # 3. Loop de Paginação: Busca os dados em lotes até chegar na data atual.
    while start_timestamp < exchange.milliseconds():
        try:
            # Busca um lote de dados a partir do último timestamp (UTC time zone) conhecido
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=start_timestamp)
            
            # Se a API não retornar mais dados, saímos do loop
            if not ohlcv:
                break

            all_data.extend(ohlcv)
            
            # Atualiza o timestamp de início para a última candlestick recebido + 1
            # para evitar buscar dados duplicados na próxima iteração.
            last_timestamp = ohlcv[-1][0]
            start_timestamp = last_timestamp + 1
            
            # Feedback para o usuário sobre o progresso
            last_date_readable = datetime.fromtimestamp(last_timestamp / 1000).strftime('%Y-%m-%d')
            print(f"   Lote recebido. Última data: {last_date_readable}")

            # Respeita o limite de requisições da API para evitar bloqueios
            exchange.sleep(exchange.rateLimit / 1000)

        except ccxt.NetworkError as e:
            print(f"Erro de rede: {e}. Tentando novamente...")
            exchange.sleep(5) # Espera 5 segundos antes de tentar novamente
        except ccxt.ExchangeError as e:
            print(f"Erro da exchange: {e}")
            return pd.DataFrame() # Encerra em caso de erro da exchange (ex: símbolo inválido)

    print("Download concluído. Processando dados...")

    if not all_data:
        print("Nenhum dado foi baixado.")
        return pd.DataFrame()

    # 4. Converte a lista de listas em um DataFrame do Pandas
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # 5. Converte a coluna de timestamp para um formato de data legível e a define como índice
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Garante que os tipos de dados estão corretos (float para OHLC e volume)
    df = df.astype(float)

    print("Processamento finalizado com sucesso.")
    return df

if __name__ == '__main__':
    import os
    from pathlib import Path
    # Bloco de teste para executar o script diretamente
    
    # Parâmetros de teste
    TEST_SYMBOL = 'BTC/USDT'
    TEST_START_DATE = '2022-01-01'
    TEST_TIMEFRAME = '1h'
    ROOT = Path(__file__).resolve().parent.parent  # sobe um nível acima de /src


    # Chama a função
    btc_data = fetch_ohlcv_data(
        symbol=TEST_SYMBOL,
        start_date=TEST_START_DATE,
        timeframe=TEST_TIMEFRAME
    )

    # Exibe os resultados do teste se o download for bem-sucedido
    if not btc_data.empty:
        print("\n--- Amostra dos Dados Baixados ---")
        print(btc_data.head())
        print("\n" + "-"*35)
        print(btc_data.tail())
        print("\n" + "-"*35)
        print(f"Dimensões do DataFrame: {btc_data.shape}")
        print(f"Período dos dados: de {btc_data.index.min()} até {btc_data.index.max()}")

        try:
            from pathlib import Path
            # Caminho da raiz do projeto
            DATA_RAW = ROOT / "data" / "raw"
            DATA_RAW.mkdir(parents=True, exist_ok=True)
            OUT_PATH = DATA_RAW / "test.csv"
            btc_data.to_csv(OUT_PATH, index=False)

            print("Working directory:", os.getcwd())

        except Exception as e:
            print("Um erro ocorreu ao tentar salvar o arquivo: ", e)