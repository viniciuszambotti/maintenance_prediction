import pandas as pd

class DataPrep:
    
    def __init__(self, df):
        self.df = df
    
    
    '''
        Clacula RUL e cria labels para fazer classificação e regressão 
        
        params
            cycles (int): número de ciclos antecipados para prevenção 
    '''
    
    
    def add_labels(self, df, cycles, is_train_data):
        
        # obtém o último ciclo de cada máquina
        df_max_cycle = pd.DataFrame(df.groupby('asset_id')['runtime'].max())
        df_max_cycle.reset_index(level=0, inplace=True)
        df_max_cycle.columns = ['asset_id', 'last_cycle']
        
        # calcula o rul
        df = pd.merge(df, df_max_cycle, on='asset_id')
        
        if is_train_data:
            df['rul'] = df['last_cycle'] - df['runtime']
        
            #label para classificação binária 
            df['failure_label'] = df['rul'].apply(lambda x: 1 if x <= cycles else 0)
        
        else:
            df = df[df['runtime'] == df['last_cycle']]
        
        df.drop(['last_cycle'], axis=1, inplace=True)


        return df
    

    '''
        Uma função para suavizar os dados do sensor afim de eliminar ruídos
        Utiliza uma janela deslizante fixa e aplicando média móvel e desvio padrão
        
        params
            window (int): Tamanho da janela para deslizar no dataset
    '''
    def smooth_data(self, window):
        col_sensor = ['t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11',
                      't12','t13','t14','t15','t16','t17','t18','t19','t20','t21']
        
        sensor_mean = [col.replace('t', 'av') for col in col_sensor]
        sensor_sdv = [col.replace('t', 'sd') for col in col_sensor]
        
        df_response = pd.DataFrame()
                
        for asset_id in pd.unique(self.df['asset_id']):
    
            # subset dos dados de sensor de um asset Id
            df_asset = self.df[self.df['asset_id'] == asset_id]
            df_asset_sub = df_asset[col_sensor]


            # Desliza no subset e calcula média
            mn = df_asset_sub.rolling(window, min_periods=1).mean()
            mn.columns = sensor_mean

            # desliza no subset e calcula desvio padrão
            sdv = df_asset_sub.rolling(window, min_periods=1).std().fillna(0)
            sdv.columns = sensor_sdv

            # combina os datasets
            new_df = pd.concat([df_asset, mn, sdv], axis=1)

            df_response = pd.concat([df_response,new_df])
                        
        return df_response
   