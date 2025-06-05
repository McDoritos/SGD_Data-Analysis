import pandas as pd
import numpy as np

def read_csv(file_name):
    df = pd.read_csv(file_name)
    print(f"File {file_name} read with success")
    return df

def read_xlsx(file_name, spreadsheet):
    if spreadsheet is None:
        df = pd.read_excel(file_name)
    else:
        df = pd.read_excel(file_name, sheet_name=spreadsheet)
            
    print(f"File {file_name} read with success")
    return df

def create_dataset_countries(df_fda,df_fde,df_gdpg,df_gdpe,df_bk,df_mac,df_obea,df_obee,df_paa,df_pae,df_fc,df_pope):
    # df_treated must have fooddiet, country, gdp, bk, mac, obesity and physicalactivity per fooddiet
    #print(df_fda.columns)

    #unique_values = {
    #    'Sex': df_fda['Sex'].unique(),
    #    'Age Group': df_fda['Age Group'].unique(),
    #    'Race and Hispanic Origin': df_fda['Race and Hispanic Origin'].unique(),
    #    'Nutrient': df_fda['Nutrient'].unique()
    #}

    #for col, values in unique_values.items():
    #    print(f"\n{col}:")
    #    for value in values:
    #        print(f" - {value}")

    #print(df_fde.columns)

    #unique_values = {
    #    'Food class level 1': df_fc['Food class level 1'].unique()
    #}
#
    #for col, values in unique_values.items():
    #    print(f"\n{col}:")
    #    for value in values:
    #        print(f" - {value}")
#
    #unique_values = {
    #    'Country': df_fde['Country'].unique(),
    #    'Pop Class': df_fde['Pop Class'].unique(),
    #    'Foodex L1': df_fde['Foodex L1'].unique()
    #}
#
    #for col, values in unique_values.items():
    #    print(f"\n{col}:")
    #    for value in values:
    #        print(f" - {value}")

    # -------------------------------
    # 1. Seleção de nutrientes americanos para o dataset americano (df_fda)
    # -------------------------------
    df_fda_treated = df_fda[
        (df_fda['Survey Years'] == '2017-2018') &
        (df_fda['Age Group'] == '2 and over') &
        (df_fda['Sex'] == 'All') &
        (df_fda['Race and Hispanic Origin'] == 'All')
    ]
    df_fda_treated = df_fda_treated[['Sex','Age Group','Nutrient', 'Mean']]

    print("Dados americanos de food diet tratados (df_fda_treated):")
    print(df_fda_treated)

    # -------------------------------
    # 2. Seleção de surveys europeus relevantes (df_fde)
    # -------------------------------
    selected_surveys = [
        'Austrian Study on Nutritional Status 2010-12 - Adults',
        'Diet National 2004',
        'National Survey of Food Intake and Nutrition',
        'Childhealth',
        'Czech National Food Consumption Survey',
        'The Danish National Dietary survey 2005-2008',
        'National Dietary Survey 1997',
        'National FINDIET 2012 Survey',
        'Individual and national study on food consumption 2',
        'National Nutrition Survey II',
        'Diet Lactation GR',
        'National Repr Surv',
        'National Adult Nutrition Survey',
        'Italian National Food Consumption Survey INRAN-SCAI 2005-06',
        'National Dietary Survey',
        'Dutch National food consumption survey 2007 - 2010',
        'National Food and Nutrition Institute - FAO 2000',
        'Dieta Pilot Adults',
        'SK MON 2008',
        'CRP-2008',
        'Spanish Agency for Food Safety (AESAN) Survey',
        'Swedish National Dietary Survey - Riksmaten adults 2010-11',
        'National Diet and Nutrition Survey - Years 1-3'
    ]
    df_selected_surveys = df_fde[df_fde['Survey'].isin(selected_surveys)]

    # -------------------------------
    # 3. Filtragem e agregação dos dados europeus por país e Foodex L1
    # -------------------------------
    df_filtered = df_selected_surveys[['Country', 'Pop Class', 'Foodex L1', 'Mean']]

    # Agrupar para calcular média por país e Foodex L1
    df_all_class = (
        df_filtered
        .groupby(['Country', 'Foodex L1'], as_index=False)
        .agg({'Mean': 'mean'})
    )

    # Definir Pop Class como 'All' para o dataframe agregado
    df_all_class['Pop Class'] = 'All'

    # Reordenar colunas para manter padrão
    df_all_class = df_all_class[['Country', 'Pop Class', 'Foodex L1', 'Mean']]

    # Concatenar dados originais filtrados com agregados (Pop Class = 'All')
    df_fde_treated = pd.concat([df_filtered, df_all_class], ignore_index=True)

    # Filtrar apenas Pop Class 'All'
    df_fde_treated = df_fde_treated[df_fde_treated['Pop Class'] == 'All']

    print("Dados europeus de food diet tratados (df_fde_treated):")
    print(df_fde_treated)

    # -------------------------------
    # 5. Filtragem e tratamentos de datasets com GDP growth
    # -------------------------------
    print(df_gdpe.columns)
    unique_values = {
        'Country': df_gdpe['Country Name'].unique()
    }
#
    for col, values in unique_values.items():
        print(f"\n{col}:")
        for value in values:
            print(f" - {value}")

    df_gdpe = df_gdpe[['Country Name', 2017]]

    df_gdpg = df_gdpg[
        (df_gdpg['Country Name'] == 'United States')
    ]
    df_gdpg = df_gdpg[['Country Name', '2017']]
    print("Dados globais de GDP tratados:")
    print(df_gdpe)
    print(df_gdpg)

    # -------------------------------
    # 6. Tratamento de datasets de restaurantes Fast food
    # -------------------------------
    print("Dados globais de restaurantes FF tratados:")
    print(df_bk)
    df_mac = df_mac[['Country','Number of MacDonald restaurants']]
    print(df_mac)

    common_countries = set(df_bk['Country']).intersection(set(df_mac['Country']))
    common_countries = sorted(list(common_countries))

    df_common_ff_restaurants = pd.DataFrame({'Country': common_countries})

    bk_counts = df_bk.set_index('Country')['Number of Burger King Restaurants']
    df_common_ff_restaurants['BurgerKing_Count'] = df_common_ff_restaurants['Country'].map(bk_counts)

    mac_counts = df_mac.set_index('Country')['Number of MacDonald restaurants']
    df_common_ff_restaurants['McDonalds_Count'] = df_common_ff_restaurants['Country'].map(mac_counts)

    print("Datasets de restaurantes FF merged:")
    print(df_common_ff_restaurants)

    # -------------------------------
    # 7. Tratamento de datasets de obesidade
    # -------------------------------
    df_obee = df_obee[['Country','Overweight']]
    print(df_obee)
    df_obea = df_obea[
        (df_obea['Survey Years'] == '2017-2018') &
        (df_obea['Age Group'] == '20 and over') &
        (df_obea['Sex'] == 'All') &
        (df_obea['Measure'] == 'Obesity') &
        (df_obea['Race and Hispanic Origin'] == 'All')
    ]
    df_obea = df_obea[['Sex','Age Group','Measure','Percent']]
    print(df_obea)

    # -------------------------------
    # 8. Tratamento de datasets de Atividade fisica
    # -------------------------------
    df_pae = df_pae[['Country','At least 4 times']]
    print(df_pae)
    df_paa = df_paa[
        (df_paa['YearStart'] == 2017)
    ]
    df_paa = df_paa[['YearStart','Class','LocationDesc','Data_Value']]
    print(df_paa)

    # -------------------------------
    # 9. Merge dos dados tratados num só dataset para a europa
    # -------------------------------
    all_countries = set()
    all_countries.update(df_fde_treated['Country'].unique())
    all_countries.update(df_gdpe['Country Name'].unique())
    all_countries.update(df_common_ff_restaurants['Country'].unique())
    all_countries.update(df_obee['Country'].unique())
    all_countries.update(df_pae['Country'].unique())

    df_final = pd.DataFrame({'Country': sorted(list(all_countries))})

    # 2. Pivotar dados de consumo alimentar
    food_consumption_pivoted = df_fde_treated.pivot_table(
        index='Country',
        columns='Foodex L1',
        values='Mean',
        aggfunc='mean'
    ).reset_index()

    # Limpar nomes das colunas
    food_consumption_pivoted.columns = [
        f"Consumo_{col.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}" 
        if col != 'Country' else col 
        for col in food_consumption_pivoted.columns
    ]

    # ARREDONDAR OS VALORES PARA 2 CASAS DECIMAIS
    # Identificar colunas de consumo (todas exceto 'Country')
    consumo_cols = [col for col in food_consumption_pivoted.columns if col != 'Country']
    food_consumption_pivoted[consumo_cols] = food_consumption_pivoted[consumo_cols].round(2)

    # Juntar ao DataFrame final
    df_final = pd.merge(df_final[['Country']],
                    food_consumption_pivoted,
                    on='Country',
                    how='left')

    # 4. Juntar dados de PIB
    df_final = pd.merge(df_final, 
                    df_gdpe.rename(columns={'Country Name': 'Country', 2017: 'GDP_2017'}),
                    on='Country',
                    how='left')

    # 5. Juntar dados de restaurantes fast-food
    df_final = pd.merge(df_final,
                    df_common_ff_restaurants,
                    on='Country',
                    how='left')

    # 6. Juntar dados de obesidade (Europa)
    df_final = pd.merge(df_final,
                    df_obee.rename(columns={'Overweight': 'Obesity_Rate_Europe'}),
                    on='Country',
                    how='left')

    # 7. Juntar dados de atividade física
    df_final = pd.merge(df_final,
                    df_pae.rename(columns={'At least 4 times': 'Physical_Activity_4plus_times'}),
                    on='Country',
                    how='left')

    # 8. Adicionar referência de obesidade dos EUA
    us_obesity = df_obea['Percent'].values[0] if len(df_obea) > 0 else None
    df_final['US_Obesity_Reference'] = us_obesity

    # 9. ADIÇÃO DOS DADOS POPULACIONAIS COMPLETOS
    df_pope_clean = df_pope[['Country', 'Population Density', 'Population', 'Area']].copy()

    # Padronizar nomes de países (opcional, mas recomendado)
    df_pope_clean['Country'] = df_pope_clean['Country'].str.strip().str.title()
    df_final['Country'] = df_final['Country'].str.strip().str.title()

    # Juntar dados populacionais completos
    df_final = pd.merge(df_final,
                    df_pope_clean,
                    on='Country',
                    how='left')

    # Renomear colunas para maior clareza (opcional)
    df_final = df_final.rename(columns={
        'Population Density': 'Pop_Density',
        'Area': 'Area_km2'
    })

    # Reordenar colunas (opcional)
    first_cols = ['Country', 'Population', 'Pop_Density', 'Area_km2']
    other_cols = [col for col in df_final.columns if col not in first_cols]
    df_final = df_final[first_cols + other_cols]

    # Ordenar e limpar
    df_final = df_final.sort_values('Country').reset_index(drop=True)

    # Verificar resultado
    print("Dataset consolidado final com todos os dados populacionais:")
    print(df_final.head())
    df_final.to_csv('consolidated_data_by_country.csv', index=False)

    return df_final

def create_features(df_final):
    cols_to_convert = ['BurgerKing_Count', 'McDonalds_Count', 'Population', 'Area_km2']

    for col in cols_to_convert:
        if col in df_final.columns:
            # Converter para string primeiro para lidar com formatação (como vírgulas decimais)
            df_final[col] = pd.to_numeric(df_final[col].astype(str).str.replace(',', '.'), errors='coerce')
        else:
            print(f"Aviso: Coluna {col} não encontrada no DataFrame")

    # 2. Calcular as métricas de fast food
    if all(col in df_final.columns for col in cols_to_convert):
        missing_values = df_final[cols_to_convert].isnull().sum()
        if missing_values.any():
            print("\nValores faltantes após conversão:")
            print(missing_values)

        # Cálculos
        df_final['Total_FastFood_Count'] = df_final['BurgerKing_Count'].fillna(0) + df_final['McDonalds_Count'].fillna(0)

        with np.errstate(divide='ignore', invalid='ignore'):
            df_final['FastFood_per_100k'] = np.where(
                df_final['Population'] > 0,
                (df_final['Total_FastFood_Count'] / df_final['Population']) * 100000,
                np.nan
            )

            df_final['FastFood_density_per_1000km2'] = np.where(
                df_final['Area_km2'] > 0,
                (df_final['Total_FastFood_Count'] / df_final['Area_km2']) * 1000,
                np.nan
            )

            df_final['BK_vs_McD_ratio'] = np.where(
                df_final['McDonalds_Count'] > 0,
                df_final['BurgerKing_Count'] / df_final['McDonalds_Count'],
                np.nan
            )

            df_final['McDonalds_percentage'] = np.where(
                df_final['Total_FastFood_Count'] > 0,
                (df_final['McDonalds_Count'] / df_final['Total_FastFood_Count']) * 100,
                np.nan
            )

        # 🔢 Arredondar os valores numéricos calculados
        round_cols = ['FastFood_per_100k', 'FastFood_density_per_1000km2',
                    'BK_vs_McD_ratio', 'McDonalds_percentage']
        df_final[round_cols] = df_final[round_cols].round(2)

        # Reordenar colunas
        ff_cols = ['Country', 'Population', 'Area_km2',
                'BurgerKing_Count', 'McDonalds_Count', 'Total_FastFood_Count'] + round_cols
        other_cols = [col for col in df_final.columns if col not in ff_cols]
        df_final = df_final[ff_cols + other_cols]

        # Mostrar estatísticas
        print("\nResumo das métricas calculadas:")
        print(df_final[round_cols].describe())
    else:
        print("Aviso: Não foi possível calcular as métricas - colunas necessárias ausentes")

    # 3. Salvar o dataset
    df_final.to_csv('consolidated_data_by_country.csv', index=False)
    return df_final

foodDiet_america = read_csv("Datasets/Treated/Food-diet_America.csv")
foodDiet_europe = read_xlsx("Datasets/Treated/Food-diet_Europe.xlsx","L1_Consuming_days_only_g_day_bw")
foodComposition = read_csv("Datasets/Treated/Food-Composition_FoodexL1.csv")

gdp_global = read_csv("Datasets/Treated/GDP-Growth_Countries.csv")
gdp_europe = read_xlsx("Datasets/Treated/GDP-Growth_Europe.xlsx",'Folha1')

bk_global = read_csv("Datasets/Treated/Num-BKs_Countries.csv")
mac_global = read_csv("Datasets/Treated/Num-MACs_Countries.csv")

obesity_america = read_csv("Datasets/Treated/Obesity_America.csv")
obesity_europe = read_xlsx("Datasets/Treated/Obesity_Europe.xlsx","Folha1")

physicalActivity_america = read_xlsx("Datasets/Treated/Physical-Activity_America.xlsx","Nutrition__Physical_Activity__a")
physicalActivity_europe = read_xlsx("Datasets/Treated/Physical-Activity_Europe.xlsx","Folha1")

Population_europe = read_csv("Datasets/Treated/Population_Europe.csv")

df_treated = create_dataset_countries(foodDiet_america,foodDiet_europe,gdp_global,gdp_europe,bk_global,mac_global,obesity_america,obesity_europe,physicalActivity_america,physicalActivity_europe,foodComposition,Population_europe)

df_final = create_features(df_treated)

print(df_final)
